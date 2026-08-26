from langgraph.graph import StateGraph, START, END
from src.pipeline.state import TAOState
from src.safety.crisis import crisis_gate
from src.agents.intake import intake_agent
from src.agents.diagnostic import diagnostic_agent
from src.agents.oversight import oversight_agent
from src.agents.routing import routing_agent
from src.agents.inconclusive import inconclusive_agent


def route_after_crisis(state: TAOState) -> str:
    """Acute risk short-circuits the entire pipeline before any LLM is invoked."""
    if state.get("crisis_escalation"):
        return END
    return "intake"


def route_after_oversight(state: TAOState) -> str:
    """Approved assessments proceed to routing; rejected ones get a safe
    inconclusive output rather than ending silently."""
    if state.get("assessment_approved"):
        return "routing"
    return "inconclusive"


def build_graph() -> StateGraph:
    graph = StateGraph(TAOState)

    graph.add_node("crisis_gate", crisis_gate)
    graph.add_node("intake", intake_agent)
    graph.add_node("diagnostic", diagnostic_agent)
    graph.add_node("oversight", oversight_agent)
    graph.add_node("routing", routing_agent)
    graph.add_node("inconclusive", inconclusive_agent)

    graph.add_edge(START, "crisis_gate")
    graph.add_conditional_edges("crisis_gate", route_after_crisis, {
        "intake": "intake",
        END: END,
    })
    graph.add_edge("intake", "diagnostic")
    graph.add_edge("diagnostic", "oversight")
    graph.add_conditional_edges("oversight", route_after_oversight, {
        "routing": "routing",
        "inconclusive": "inconclusive",
    })
    graph.add_edge("routing", END)
    graph.add_edge("inconclusive", END)

    return graph.compile()


tao_graph = build_graph()
