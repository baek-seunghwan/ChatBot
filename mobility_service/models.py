from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class OrderType(str, Enum):
    QUICK = "QUICK"
    QUICK_ECONOMY = "QUICK_ECONOMY"
    QUICK_EXPRESS = "QUICK_EXPRESS"
    DOBO = "DOBO"


class ProductSize(str, Enum):
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"


class PaymentType(str, Enum):
    CARD = "CARD"
    CASH_ON_PICKUP = "CASH_ON_PICKUP"
    CASH_ON_DROPOFF = "CASH_ON_DROPOFF"


class Location(CamelModel):
    basic_address: str = Field(alias="basicAddress", min_length=2, max_length=200)
    detail_address: str | None = Field(
        default=None, alias="detailAddress", max_length=200
    )
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Contact(CamelModel):
    name: str = Field(min_length=1, max_length=50)
    phone: str = Field(pattern=r"^[0-9+\-\s]{8,20}$")


class DeliveryStop(CamelModel):
    location: Location
    contact: Contact | None = None
    note: str | None = Field(default=None, max_length=500)


class DeliveryDraft(CamelModel):
    order_type: OrderType = Field(default=OrderType.QUICK, alias="orderType")
    product_size: ProductSize = Field(default=ProductSize.XS, alias="productSize")
    pickup: DeliveryStop
    dropoff: DeliveryStop
    waypoints: list[DeliveryStop] = Field(default_factory=list, max_length=10)
    wish_time: str | None = Field(default=None, alias="wishTime")
    product_name: str = Field(
        default="배송 물품", alias="productName", min_length=1, max_length=100
    )
    quantity: str = Field(default="1", max_length=20)
    declared_value: int | None = Field(default=None, alias="declaredValue", ge=0)
    payment_type: PaymentType = Field(default=PaymentType.CARD, alias="paymentType")

    @model_validator(mode="after")
    def validate_constraints(self) -> "DeliveryDraft":
        if self.waypoints and self.order_type in {
            OrderType.DOBO,
            OrderType.QUICK_ECONOMY,
        }:
            raise ValueError("도보 배송과 퀵 이코노미는 경유지를 지원하지 않습니다.")
        if self.waypoints and self.payment_type != PaymentType.CARD:
            raise ValueError("경유지가 있는 주문은 카드 결제만 지원합니다.")
        return self


class CreateDeliveryRequest(DeliveryDraft):
    partner_order_id: str | None = Field(
        default=None,
        alias="partnerOrderId",
        min_length=4,
        max_length=100,
        pattern=r"^[A-Za-z0-9._-]+$",
    )

    @model_validator(mode="after")
    def require_contacts(self) -> "CreateDeliveryRequest":
        missing = []
        if self.pickup.contact is None:
            missing.append("출발지")
        if self.dropoff.contact is None:
            missing.append("도착지")
        for index, waypoint in enumerate(self.waypoints, start=1):
            if waypoint.contact is None:
                missing.append(f"경유지 {index}")
        if missing:
            raise ValueError(f"{', '.join(missing)} 연락처가 필요합니다.")
        return self


class CallbackBody(CamelModel):
    picker_id: str | None = Field(default=None, alias="pickerId")
    reason: str | None = None
    cancel_by: str | None = Field(default=None, alias="cancelBy")
    cancel_fee: int | None = Field(default=None, alias="cancelFee")
    image_url: str | None = Field(default=None, alias="imageUrl")
    encoded_step_id: str | None = Field(default=None, alias="encodedStepId")


class ApiEnvelope(CamelModel):
    ok: bool = True
    data: Any = None
    message: str | None = None


class AgentChatRequest(CamelModel):
    session_id: str | None = Field(default=None, alias="sessionId")
    message: str = Field(min_length=1, max_length=1000)
    mode: str = Field(default="ai", pattern=r"^(ai|local)$")

