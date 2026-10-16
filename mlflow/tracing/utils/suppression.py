import contextlib
from contextvars import ContextVar

from opentelemetry import context as otel_context

# True while MLflow exports traces. The span processor skips root spans started under it, which
# covers instrumentation that ignores OpenTelemetry's suppression key, such as storage SDKs'
# own tracing.
_IN_TRACE_EXPORT: ContextVar[bool] = ContextVar("mlflow_in_trace_export", default=False)


@contextlib.contextmanager
def suppress_export_tracing():
    """
    Keep MLflow's trace export from recording its own I/O as traces when it shares the tracer
    provider with other OpenTelemetry instrumentation.
    """
    token = _IN_TRACE_EXPORT.set(True)
    # The same key the OpenTelemetry SDK's span processors set around `export()`.
    otel_token = otel_context.attach(
        otel_context.set_value(otel_context._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        yield
    finally:
        otel_context.detach(otel_token)
        _IN_TRACE_EXPORT.reset(token)
