"""Taiwan stock trading costs."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Defaults are the statutory rates with no broker discount.

    commission_discount: e.g. 0.28 for a 2.8折 online-trading discount.
    min_commission: many brokers charge at least NT$20 per order (some charge
        NT$1 for odd lots; set it to match your broker).
    slippage: fraction of price paid against you on every fill, a stand-in
        for the bid-ask spread and market impact at the open.
    """

    commission_rate: float = 0.001425
    commission_discount: float = 1.0
    min_commission: float = 20.0
    stock_tax_rate: float = 0.003  # 證交稅, sells only
    etf_tax_rate: float = 0.001
    slippage: float = 0.001

    def commission(self, value: float) -> float:
        if value <= 0:
            return 0.0
        fee = math.floor(value * self.commission_rate * self.commission_discount)
        return float(max(fee, self.min_commission))

    def tax(self, value: float, stock_id: str) -> float:
        if value <= 0:
            return 0.0
        rate = self.etf_tax_rate if stock_id.startswith("00") else self.stock_tax_rate
        return float(math.floor(value * rate))

    def buy_price(self, price: float) -> float:
        return price * (1 + self.slippage)

    def sell_price(self, price: float) -> float:
        return price * (1 - self.slippage)

    def buy_cost(self, shares: float, price: float) -> float:
        """Total cash needed to buy ``shares`` at quoted ``price``."""
        value = shares * self.buy_price(price)
        return value + self.commission(value)


ZERO_COST = CostModel(commission_rate=0, min_commission=0, stock_tax_rate=0, etf_tax_rate=0,
                      slippage=0)
