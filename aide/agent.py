import logging
import random
import time
from typing import Any, Callable, cast

import humanize
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("aide")


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the training script.",
)

hyperparameter_tuning_check_spec = FunctionSpec(
    name="check_hyperparameter_tuning",
    json_schema={
        "type": "object",
        "properties": {
            "has_hyperparameter_tuning": {
                "type": "boolean",
                "description": "true if the code includes hyperparameter tuning/optimization (e.g., GridSearchCV, RandomizedSearchCV, Optuna, hyperopt, manual parameter sweeps, cross-validation with parameter search, etc.), false otherwise.",
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of why hyperparameter tuning is or is not present in the code.",
            },
        },
        "required": ["has_hyperparameter_tuning", "explanation"],
    },
    description="Check if the code includes hyperparameter tuning or optimization.",
)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()  # Track start time for time-based enforcement

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] greedy node selected")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description. Alternatively, you can save it to "./best_submission/submission.csv" if that directory exists.** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        # Add hyperparameter tuning guidance if required
        # Only enforce during first 1/3 of time limit
        elapsed_time = time.time() - self.start_time
        time_limit_secs = self.acfg.time_limit if self.acfg.time_limit else float('inf')
        if self.acfg.require_hyperparameter_tuning and elapsed_time < (time_limit_secs / 3.0):
            prompt["Instructions"]["Hyperparameter tuning note"] = [
                "Note: This solution will need hyperparameter tuning in later iterations.",
                "When adding hyperparameter tuning, prefer RandomizedSearchCV (n_iter=20-50) or Optuna (n_trials=20-50).",
                "These methods are typically more effective than GridSearchCV while being faster and less likely to timeout.",
            ]
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        # Add hyperparameter tuning guidance if required and not present
        # Only enforce during first 1/3 of time limit
        elapsed_time = time.time() - self.start_time
        time_limit_secs = self.acfg.time_limit if self.acfg.time_limit else float('inf')
        if self.acfg.require_hyperparameter_tuning and elapsed_time < (time_limit_secs / 3.0):
            has_tuning, _ = self._has_hyperparameter_tuning(parent_node.code)
            if not has_tuning:
                prompt["Instructions"]["Hyperparameter tuning requirement"] = [
                    "**IMPORTANT: The previous solution is missing hyperparameter tuning.**",
                    "If improving the model, consider adding hyperparameter tuning using RandomizedSearchCV (n_iter=20-50) or Optuna (n_trials=20-50).",
                    "These methods are typically MORE effective than GridSearchCV while being faster and less likely to timeout.",
                    "Limit the search space and use 3-5 CV folds to balance speed and reliability.",
                ]
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _debug(self, parent_node: Node) -> Node:
        # Check if this is a hyperparameter tuning requirement issue
        # Only enforce during first 1/3 of time limit
        elapsed_time = time.time() - self.start_time
        time_limit_secs = self.acfg.time_limit if self.acfg.time_limit else float('inf')
        is_tuning_issue = False
        if (
            self.acfg.require_hyperparameter_tuning
            and elapsed_time < (time_limit_secs / 3.0)
            and parent_node.exc_type is None
            and "Missing hyperparameter tuning" in (parent_node.analysis or "")
        ):
            is_tuning_issue = True

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        
        if is_tuning_issue:
            prompt["Instructions"]["Critical requirement"] = [
                "**The previous solution is missing hyperparameter tuning/optimization.**",
                "You MUST add systematic hyperparameter tuning to the solution. This includes:",
                "- **PREFER RandomizedSearchCV, Optuna, or hyperopt over GridSearchCV**",
                "  * RandomizedSearchCV (n_iter=20-50) is often MORE effective than GridSearchCV while being much faster",
                "  * Optuna (n_trials=20-50) uses Bayesian optimization and is typically the most effective method",
                "  * GridSearchCV can be too slow and cause timeouts, especially with large parameter spaces",
                "- Systematic exploration of hyperparameter space (learning rate, regularization, tree depth, batch size, etc.)",
                "- Cross-validation with parameter search (use 3-5 folds to balance speed and reliability)",
                "- Limit the search space to avoid timeouts - focus on the most important hyperparameters",
                "- The solution should NOT be accepted without proper hyperparameter tuning.",
            ]
        
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)

    def _has_hyperparameter_tuning(self, code: str) -> tuple[bool, str]:
        """Check if the code includes hyperparameter tuning using LLM."""
        prompt = {
            "Introduction": (
                "You are analyzing Python machine learning code to determine if it includes hyperparameter tuning or optimization. "
                "Hyperparameter tuning includes: GridSearchCV, RandomizedSearchCV, Optuna, hyperopt, manual parameter sweeps, "
                "cross-validation with parameter search, Bayesian optimization, or any systematic exploration of hyperparameter space."
            ),
            "Code": wrap_code(code),
            "Instructions": (
                "Analyze the code and determine if it includes hyperparameter tuning. "
                "Look for systematic parameter search, optimization libraries, or iterative parameter exploration."
            ),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=hyperparameter_tuning_check_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
            ),
        )

        has_tuning = response.get("has_hyperparameter_tuning", False)
        explanation = response.get("explanation", "")
        return has_tuning, explanation

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        prompt = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            ),
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
            ),
        )

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
        )

        # Check for hyperparameter tuning if required and execution succeeded
        # Disable enforcement after 1/3 of time limit to allow completion
        elapsed_time = time.time() - self.start_time
        time_limit_secs = self.acfg.time_limit if self.acfg.time_limit else float('inf')
        enforce_tuning = (
            self.acfg.require_hyperparameter_tuning 
            and elapsed_time < (time_limit_secs / 3.0)
        )
        
        if (
            enforce_tuning
            and not node.is_buggy
            and response["metric"] is not None
        ):
            has_tuning, tuning_explanation = self._has_hyperparameter_tuning(node.code)
            if not has_tuning:
                logger.info(
                    f"Node {node.id} marked as buggy due to missing hyperparameter tuning. "
                    f"Explanation: {tuning_explanation}"
                )
                node.is_buggy = True
                node.analysis = (
                    f"{node.analysis}\n\n"
                    f"**Missing hyperparameter tuning**: {tuning_explanation}"
                )
        elif (
            self.acfg.require_hyperparameter_tuning 
            and elapsed_time >= (time_limit_secs / 3.0)
            and not hasattr(self, '_tuning_disabled_logged')
        ):
            logger.info(
                f"Hyperparameter tuning enforcement disabled after {elapsed_time:.1f}s "
                f"(1/3 of {time_limit_secs}s time limit) to allow completion"
            )
            self._tuning_disabled_logged = True

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
