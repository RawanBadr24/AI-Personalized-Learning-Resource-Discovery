"""Builds the agent team: an AgentWorkflow with handoffs.

    InfoAgent ──► SearchAgent ◄──► EvaluationAgent ──► LearningPathAgent
                  (loop, capped at 3 cycles by the tools)   (ranking runs inside finalize_evaluation)
"""
from learning_agent.agents import tools as T
from learning_agent.prompts import EVAL_AGENT, INFO_AGENT, PATH_AGENT, SEARCH_AGENT
from learning_agent.tools import profile as profile_tools


def build_team(user_id: str, llm=None):
    from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
    from llama_index.core.tools import FunctionTool

    from learning_agent.llm import get_llm

    llm = llm or get_llm()

    def tools(*pairs):
        # pairs: fn or (fn, public_name)
        out = []
        for p in pairs:
            fn, name = p if isinstance(p, tuple) else (p, None)
            out.append(FunctionTool.from_defaults(fn=fn, name=name))
        return out

    info = FunctionAgent(
        name="InfoAgent",
        description="Talks to the user, builds and saves the structured learning profile.",
        system_prompt=INFO_AGENT.format(user_id=user_id),
        llm=llm,
        tools=tools(
            profile_tools.profile_retrieve,
            profile_tools.analyze_learning_goal,
            profile_tools.assess_user_level_and_preferences,
            profile_tools.goal_decomposer,
            (T.save_profile, "data_save"),
        ),
        can_handoff_to=["SearchAgent"],
    )
    search = FunctionAgent(
        name="SearchAgent",
        description="Writes search queries and discovers learning resources on free platforms.",
        system_prompt=SEARCH_AGENT,
        llm=llm,
        tools=tools(T.get_learning_brief, T.search_resources),
        can_handoff_to=["EvaluationAgent"],
    )
    evaluation = FunctionAgent(
        name="EvaluationAgent",
        description="Scores resources, checks topic coverage, and decides: search again or finalize.",
        system_prompt=EVAL_AGENT,
        llm=llm,
        tools=tools(
            T.get_learning_brief, T.get_pending_resources, T.record_evaluations, T.check_topic_coverage,
            T.find_in_index, T.request_more_search, T.finalize_evaluation,
        ),
        can_handoff_to=["SearchAgent", "LearningPathAgent"],
    )
    path = FunctionAgent(
        name="LearningPathAgent",
        description="Orders topics by prerequisites and builds the personalized roadmap with practice tasks.",
        system_prompt=PATH_AGENT,
        llm=llm,
        tools=tools(T.get_learning_brief, T.get_ranked_resources, T.find_in_index, T.save_roadmap),
    )
    return AgentWorkflow(agents=[info, search, evaluation, path], root_agent="InfoAgent", timeout=None)
