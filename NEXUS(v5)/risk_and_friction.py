import numpy as np
from config_and_data import SystemContext

# ==========================================
# TRANSACTION FRICTION ROUTERS
# ==========================================
def apply_entry_friction(price, ctx: SystemContext):
    """Apply slippage/spread when entering a position."""
    return price * (1 + ctx.SLIPPAGE_ENTRY)

def apply_exit_friction(price, ctx: SystemContext):
    """Apply slippage/spread when exiting a position."""
    return price * (1 - ctx.SLIPPAGE_EXIT)

def apply_cash_drag(cash, ctx: SystemContext):
    """Apply daily risk-free rate to uninvested cash."""
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
# VOLATILITY-ADJUSTED POSITION SIZER (Corrected)
# ==========================================
def position_size(entry_price, stop_price, equity, ctx: SystemContext, override_alloc_fraction=None):
    """
    Sizes execution shares based on risk parameters.
    
    IMPORTANT: entry_price should already have friction applied!
    This was the critical bug - friction was being applied after sizing,
    which meant position sizing used one price but actual entry was at another.
    
    Args:
        entry_price: Friction-adjusted entry price (not raw open!)
        stop_price: Stop loss price (in terms of adjusted entry)
        equity: Available portfolio equity
        ctx: SystemContext with risk parameters
        override_alloc_fraction: Optional dynamic allocation override
    
    Returns:
        Number of shares to enter (integer)
    """
    # Calculate risk distance in actual dollars
    risk_dist = entry_price - stop_price
    if risk_dist <= 0:
        return 0
    
    # Size based on risk per trade
    risk_dollars = (equity * ctx.RISK_PER_TRADE) - ctx.COMMISSION
    if risk_dollars <= 0:
        return 0
    
    shares_risk = risk_dollars / risk_dist
    
    # Size based on maximum allocation fraction
    alloc_fraction = override_alloc_fraction if override_alloc_fraction is not None \
                     else ctx.MAX_ALLOC_FRACTION
    
    max_alloc_dollars = (equity * alloc_fraction) - ctx.COMMISSION
    if max_alloc_dollars <= 0:
        return 0
    
    shares_alloc = max_alloc_dollars / entry_price
    
    # Return the more conservative sizing
    final_shares = min(shares_risk, shares_alloc)
    return int(final_shares) if final_shares > 0 else 0

# ==========================================
# SECTOR DENSITY CROWDING CIRCUITS (Corrected)
# ==========================================
def sector_cap_reached(open_positions, sector, ctx: SystemContext, override_max_sector=None):
    """
    Evaluates whether adding another position in this sector would violate
    sector concentration limits.
    
    Args:
        open_positions: Dict of currently open positions
        sector: Sector string to check
        ctx: SystemContext with sector limits
        override_max_sector: Optional dynamic sector cap override
    
    Returns:
        True if adding another position in this sector would exceed limits
    """
    # Count existing positions in this sector
    count = sum(1 for p in open_positions.values() if p.get('sector') == sector)
    
    # Use override if provided, otherwise use context limit
    max_sector = override_max_sector if override_max_sector is not None \
                 else ctx.MAX_SECTOR_POSITIONS
    
    return count >= max_sector


# ==========================================
# UTILITY FUNCTIONS FOR DEBUGGING
# ==========================================
def validate_position_calc(entry_price, stop_price, equity, shares, commission, slippage_entry):
    """
    Verify that position sizing makes sense.
    Useful for debugging.
    """
    if shares <= 0:
        return True  # No position taken, nothing to validate
    
    # Actual entry cost with friction and commission
    actual_entry = entry_price * (1 + slippage_entry)
    total_cost = (shares * actual_entry) + commission
    
    # Risk in dollars
    risk_dollars = shares * (entry_price - stop_price)
    
    # Allocation in dollars
    alloc_dollars = shares * entry_price
    
    return {
        'entry_price': entry_price,
        'shares': shares,
        'total_cost_with_friction': total_cost,
        'risk_dollars': risk_dollars,
        'alloc_percent': (alloc_dollars / equity) if equity > 0 else 0.0
    }