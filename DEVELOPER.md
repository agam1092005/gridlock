# Developer Guide

Welcome to the Gridlock 2.0 codebase! 

## Local Setup

1. Clone the repository.
2. Install dependencies via Poetry: `poetry install`
3. We utilize standard Python tooling. Ensure your IDE is configured for `black` formatting and `flake8` linting.

## Configuration

Gridlock 2.0 relies on `pydantic-settings` to manage global variables.
- Default settings live in `src/config/config.yaml`.
- You can override any of these by passing environment variables. E.g., `export API_PORT=8080`.

### Extensibility

**Adding new Playbook Rules:**
Do not edit the `PlaybookEngine` code. Instead, open `src/config/playbooks.yaml` and append your new rules under the desired `incident_type`.

**Adding new Metrics:**
Import `metrics_registry` from `src/monitoring/metrics.py` and simply call `metrics_registry.inc_counter('my_new_metric')` or `observe_histogram`.

## Troubleshooting

- **Redis/DB Down Errors:** Since this is a lightweight prototype, we mock the databases in the health endpoint.
- **Missing Module Warning:** If `uvicorn` fails to find `src`, ensure you run it using `poetry run` from the absolute root of the project directory.
