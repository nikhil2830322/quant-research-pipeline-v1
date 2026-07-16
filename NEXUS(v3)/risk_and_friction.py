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
# VOLATILITY-ADJUSTED POSITION SIZER (Ensemble Adaptive)
# ==========================================
def position_size(entry_price, stop_price, equity, ctx: SystemContext, override_alloc_fraction=None):
    """
    Sizes execution shares. Dynamically accepts dynamic fraction parameters 
    passed down by the multi-strategy master execution pipeline.
    """
    friction_entry = apply_entry_friction(entry_price, ctx)
    risk_dist = friction_entry - stop_price
    if risk_dist <= 0:
        return 0
        
    risk_dollars = (equity * ctx.RISK_PER_TRADE) - ctx.COMMISSION
    shares_risk = risk_dollars / risk_dist
    
    # FIXED: Allow the Ensemble to override allocation fractions dynamically
    alloc_fraction = override_alloc_fraction if override_alloc_fraction is not None else ctx.MAX_ALLOC_FRACTION
    
    max_alloc = (equity * alloc_fraction) - ctx.COMMISSION
    shares_alloc = max_alloc / friction_entry
    return int(min(shares_risk, shares_alloc))

# ==========================================
# SECTOR DENSITY CROWDING CIRCUITS (Ensemble Adaptive)
# ==========================================
def sector_cap_reached(open_positions, sector, ctx: SystemContext, override_max_sector=None):
    """
    Evaluates sector concentration density across open positions.
    """
    count = sum(1 for p in open_positions.values() if p['sector'] == sector)
    
    # FIXED: Allow the Ensemble to pass down dynamic concentration bands based on active regimes
    max_sector = override_max_sector if override_max_sector is not None else ctx.MAX_SECTOR_POSITIONS
    return count >= max_sector
