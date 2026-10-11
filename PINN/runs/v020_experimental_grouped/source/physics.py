"""Explicitly gated LLD prior for a dry mobile layer, never bonded adsorption."""
import numpy as np
import pandas as pd
import torch

LLD_COLUMNS = ['Withdrawal Velocity (mm/s)', 'Viscosity (cP)',
               'Surface Tension (mN/m)', 'Solution Density (kg/m3)',
               'Wet-to-Dry Mobile Factor', 'LLD Applicable']


def validate_physics(df, cfg):
    if cfg.get('velocity_smooth_weight', 0):
        col = cfg['velocity_column']
        if col not in cfg['features'] or col not in df:
            raise ValueError('Velocity smoothness requires measured velocity as an input.')
        if df[col].nunique() < 2 or (df[col] <= 0).any():
            raise ValueError('Velocity smoothness requires varying positive measured velocities.')
    if not cfg.get('hydrodynamic_weight', 0):
        return
    missing = set(LLD_COLUMNS) - set(df)
    if missing:
        raise ValueError(f'LLD enabled but measured physics columns missing: {sorted(missing)}')
    values = df[LLD_COLUMNS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('LLD physics fields must be finite.')
    if not values['LLD Applicable'].isin([0, 1]).all():
        raise ValueError('LLD Applicable must explicitly be 0 or 1.')
    active = values.loc[values['LLD Applicable'] == 1]
    if (active[LLD_COLUMNS[:-1]] <= 0).any().any():
        raise ValueError('Active LLD measurements and conversion factors must be positive.')
    if (active['Wet-to-Dry Mobile Factor'] > 1).any():
        raise ValueError('Wet-to-dry mobile factor must lie in (0,1].')
    if not active.empty:
        ca = active['Viscosity (cP)']*1e-3 * active['Withdrawal Velocity (mm/s)']*1e-3 / (active['Surface Tension (mN/m)']*1e-3)
        if (ca >= cfg.get('lld_max_capillary_number', .01)).any():
            raise ValueError('Active LLD rows exceed the configured small-capillary-number regime.')


def mobile_lld_nm(df):
    """Flat plate, Newtonian, clean interface, low Ca; validity is lab supplied.

    h_wet = 0.9458 sqrt(gamma/(rho*g)) (eta*U/gamma)^(2/3).
    The explicit per-row dry conversion is supplied, never inferred from target.
    """
    eta = df['Viscosity (cP)'].to_numpy(float)*1e-3
    speed = df['Withdrawal Velocity (mm/s)'].to_numpy(float)*1e-3
    gamma = df['Surface Tension (mN/m)'].to_numpy(float)*1e-3
    rho = df['Solution Density (kg/m3)'].to_numpy(float)
    return .9458*np.sqrt(gamma/(rho*9.81))*(eta*speed/gamma)**(2/3)*1e9*df['Wet-to-Dry Mobile Factor'].to_numpy(float)


def hydrodynamic_loss(pred, total, df, output_scale, cfg):
    if not cfg.get('hydrodynamic_weight', 0):
        return pred.new_zeros(())
    active = df['LLD Applicable'].to_numpy() == 1
    if not active.any():
        return pred.new_zeros(())
    expected = torch.tensor(mobile_lld_nm(df.loc[active]), dtype=pred.dtype, device=pred.device).reshape(-1, 1)
    mask = torch.tensor(active, device=pred.device)
    return (((total[mask]-pred[mask])-expected)/output_scale).square().mean()
