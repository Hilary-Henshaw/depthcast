"""Render the DepthCast architecture diagram to assets/architecture.png.

This script is app-independent: it reads nothing but the known module
layout and emits a PNG.

Primary renderer: the `diagrams` library (which shells out to Graphviz).
Install it with::

    pip install diagrams        # also needs the Graphviz `dot` binary

Fallback renderer: if `diagrams` is not installed, the script builds the
same graph as Graphviz DOT and renders it with the `dot` binary directly,
so the asset is still produced.

Run::

    python scripts/architecture_diagram.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_OUTPUT_STEM = _ASSETS / "architecture"
_OUTPUT_PNG = _ASSETS / "architecture.png"


def render_with_diagrams() -> bool:
    """Render using the `diagrams` library. Return True on success."""
    try:
        from diagrams import Cluster, Diagram, Edge
        from diagrams.generic.blank import Blank
        from diagrams.generic.storage import Storage
        from diagrams.programming.language import Python
    except ImportError:
        return False

    graph_attr = {"fontsize": "18", "bgcolor": "white", "pad": "0.4"}
    with Diagram(
        "DepthCast",
        filename=str(_OUTPUT_STEM),
        outformat="png",
        show=False,
        direction="LR",
        graph_attr=graph_attr,
    ):
        with Cluster("Interfaces"):
            cli = Python("Typer CLI")
            config = Blank("ExperimentConfig\n(Pydantic)")

        with Cluster("Data sources"):
            fi2010 = Storage("FI-2010 file")
            synthetic = Python("Synthetic\ngenerator")

        with Cluster("Pipeline"):
            split = Python("split_session")
            dataset = Python("LobWindowDataset")
            loader = Python("DataLoader")

        with Cluster("DepthCastNet"):
            compress = Python("Compression\nstack")
            fusion = Python("Multi-scale\nfusion")
            lstm = Python("LSTM + head")
            compress >> fusion >> lstm

        with Cluster("Training and evaluation"):
            trainer = Python("Trainer.fit")
            checkpoint = Storage("checkpoint\nbest.pt")
            metrics = Python("evaluate_\nclassifier")

        cli >> config
        [fi2010, synthetic] >> split >> dataset >> loader
        loader >> compress
        loader >> trainer
        lstm >> trainer >> checkpoint
        checkpoint >> Edge(label="reload") >> metrics
        loader >> metrics

    return True


def render_with_dot() -> bool:
    """Fallback: render the graph with the Graphviz `dot` binary."""
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        return False

    dot_source = _build_dot()
    _ASSETS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [dot_binary, "-Tpng", "-o", str(_OUTPUT_PNG)],
        input=dot_source,
        text=True,
        check=True,
    )
    return True


def _build_dot() -> str:
    """Return a Graphviz DOT description of the data flow."""
    return """
digraph DepthCast {
  rankdir=LR;
  bgcolor="white";
  pad=0.4;
  node [shape=box, style="rounded,filled", fontname="Helvetica",
        fillcolor="#eef3fb", color="#3b6db5", fontsize=11];
  edge [color="#5b6b7a", fontname="Helvetica", fontsize=9];

  subgraph cluster_iface {
    label="Interfaces"; color="#b5c4d8"; style="rounded";
    cli [label="Typer CLI"];
    config [label="ExperimentConfig\\n(Pydantic)", fillcolor="#fde9f0",
            color="#c25080"];
  }

  subgraph cluster_data {
    label="Data sources"; color="#b5c4d8"; style="rounded";
    fi2010 [label="FI-2010 file", shape=cylinder, fillcolor="#e7f0e7",
            color="#5a8a5a"];
    synthetic [label="Synthetic\\ngenerator"];
  }

  subgraph cluster_pipe {
    label="Pipeline"; color="#b5c4d8"; style="rounded";
    split [label="split_session"];
    dataset [label="LobWindowDataset\\n(lazy windows)"];
    loader [label="DataLoader"];
  }

  subgraph cluster_model {
    label="DepthCastNet"; color="#b5c4d8"; style="rounded";
    compress [label="Compression\\nstack"];
    fusion [label="Multi-scale\\nfusion"];
    head [label="LSTM + head\\n(logits)"];
    compress -> fusion -> head;
  }

  subgraph cluster_train {
    label="Training and evaluation"; color="#b5c4d8"; style="rounded";
    trainer [label="Trainer.fit"];
    checkpoint [label="checkpoint\\nbest.pt", shape=cylinder,
                fillcolor="#e7f0e7", color="#5a8a5a"];
    metrics [label="evaluate_classifier\\n(accuracy, F1)"];
  }

  cli -> config;
  config -> trainer [style=dashed, label="drives"];
  fi2010 -> split;
  synthetic -> split;
  split -> dataset -> loader;
  loader -> compress;
  loader -> trainer;
  head -> trainer;
  trainer -> checkpoint;
  checkpoint -> metrics [label="reload"];
  loader -> metrics;
}
"""


def main() -> None:
    """Render the diagram, preferring `diagrams`, then `dot`."""
    _ASSETS.mkdir(parents=True, exist_ok=True)
    if render_with_diagrams():
        renderer = "diagrams"
    elif render_with_dot():
        renderer = "graphviz dot (fallback)"
    else:
        raise RuntimeError(
            "Neither the 'diagrams' library nor the 'dot' binary is "
            "available. Install one: pip install diagrams (with "
            "Graphviz), or apt-get install graphviz."
        )
    print(f"Wrote {_OUTPUT_PNG} using {renderer}.")


if __name__ == "__main__":
    main()
