from langgraph.graph import StateGraph, START, END
from src.pipeline.state import TAOState
from src.agents.intake import intake_agent
from src.agents.diagnostic import diagnostic_agent
from src.agents.oversight import oversight_agent
from src.agents.routing import routing_agent


def route_after_oversight(state: TAOState) -> str:
    """
    Conditional edge: if the Oversight Agent approved the assessment,
    proceed to routing. Otherwise end with an inconclusive result.
    """
    if state.get("assessment_approved"):
        return "routing"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(TAOState)

    graph.add_node("intake", intake_agent)
    graph.add_node("diagnostic", diagnostic_agent)
    graph.add_node("oversight", oversight_agent)
    graph.add_node("routing", routing_agent)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "diagnostic")
    graph.add_edge("diagnostic", "oversight")
    graph.add_conditional_edges("oversight", route_after_oversight, {
        "routing": "routing",
        END: END,
    })
    graph.add_edge("routing", END)

    return graph.compile()


tao_graph = build_graph()
