class ProviderUnavailable(Exception):
    """The upstream model or embeddings service could not complete a request."""


class ProviderTimeout(ProviderUnavailable):
    """A local model request timed out."""


class AgentLimitReached(Exception):
    """The model exceeded the tool-call budget."""
