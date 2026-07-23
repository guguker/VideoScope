from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ProviderState(StrEnum):
    READY = "ready"
    NEEDS_CONFIGURATION = "needs_configuration"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    id: str
    label: str
    state: ProviderState
    detail: str
    optional: bool = False


class Provider(Protocol):
    @property
    def id(self) -> str: ...

    def status(self) -> ProviderStatus: ...


class StaticProvider:
    def __init__(
        self,
        provider_id: str,
        label: str,
        state: ProviderState,
        detail: str,
        *,
        optional: bool = False,
    ) -> None:
        self.id = provider_id
        self.label = label
        self.state = state
        self.detail = detail
        self.optional = optional

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            id=self.id,
            label=self.label,
            state=self.state,
            detail=self.detail,
            optional=self.optional,
        )


class ProviderRegistry:
    def __init__(self, providers: list[Provider]) -> None:
        identifiers = [provider.id for provider in providers]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate provider ID")
        self._providers = {provider.id: provider for provider in providers}

    def statuses(self) -> list[ProviderStatus]:
        return [provider.status() for provider in self._providers.values()]

    def is_ready(self, provider_id: str) -> bool:
        provider = self._providers.get(provider_id)
        return provider is not None and provider.status().state is ProviderState.READY

    def get(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

