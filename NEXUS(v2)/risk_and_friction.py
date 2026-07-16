import numpy as np
from config_and_data import SystemContext

# ==========================================
# TRANSACTION FRICTION ROUTERS
# ==========================================
def apply_entry_friction(price, ctx: SystemContext):
    return price * (1 + ctx.SLIPPAGE_ENTRY)

def apply_exit_friction(price, ctx: SystemContext):
    return price * (1 - ctx.SLIPPAGE_EXIT)

def apply_cash_drag(cash, ctx: SystemContext):
    return cash * (1 + ctx.DAILY_RF)

# ==========================================
# MICROSTRUCTURE OVERNIGHT TAIL RISK
# ==========================================
def gap_down_exit(open_price, stop_price, close_price):
    """
    If the day's open already gapped below the stop, fill at that
    (worse) gapped-down open. Otherwise hold through to that day's close
    (matches the live script's MOC exit routing).
    """
    if open_price < stop_price:
        return open_price
    return close_price

# ==========================================
# VOLATILITY-ADJUSTED POSITION SIZER
# ==========================================
def position_size(entry_price, stop_price, equity, ctx: SystemContext):
    friction_entry = apply_entry_friction(entry_price, ctx)
    risk_dist = friction_entry - stop_price
    if risk_dist <= 0:
        return 0
    risk_dollars = (equity * ctx.RISK_PER_TRADE) - ctx.COMMISSION
    shares_risk = risk_dollars / risk_dist
    max_alloc = (equity * ctx.MAX_ALLOC_FRACTION) - ctx.COMMISSION
    shares_alloc = max_alloc / friction_entry
    return int(min(shares_risk, shares_alloc))

# ==========================================
# SECTOR DENSITY CROWDING CIRCUITS
# ==========================================
def sector_cap_reached(open_positions, sector, ctx: SystemContext):
    count = sum(1 for p in open_positions.values() if p['sector'] == sector)
    return count >= ctx.MAX_SECTOR_POSITIONS
