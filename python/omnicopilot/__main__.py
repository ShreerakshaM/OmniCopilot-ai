"""CLI entry point for OmniCopilot."""

import typer

app = typer.Typer(name="omnicopilot", help="OmniCopilot CLI")


@app.command()
def main() -> None:
    """OmniCopilot — Cooperative Multimodal AI Under Communication Constraints."""
    typer.echo("OmniCopilot v0.1.0")
    typer.echo("Run 'omnicopilot --help' for available commands.")


if __name__ == "__main__":
    app()
