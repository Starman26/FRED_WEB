from pathlib import Path

# Tu grafo compilado vive aquí:
from src.agent.graph import graph, create_graph_with_verification


def export(name: str, g):
    # Mermaid (texto)
    mmd = g.get_graph().draw_mermaid()
    Path(f"{name}.mmd").write_text(mmd, encoding="utf-8")

    # PNG (imagen)
    png = g.get_graph().draw_mermaid_png()
    Path(f"{name}.png").write_bytes(png)

    print(f"OK -> {name}.mmd + {name}.png")


if __name__ == "__main__":
    export("multi_agent_graph", graph)

    # Opcional: versión con verificación
    g2 = create_graph_with_verification()
    export("multi_agent_graph_with_verification", g2)
