"""Inference from a saved neural checkpoint on an explicitly supplied table."""
import argparse
from pathlib import Path
import torch
from core import ThicknessNet, Preprocessor, load_data, predict

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    df = load_data(args.data, cfg, require_target=False)
    model = ThicknessNet(cfg, checkpoint['bounded'])
    model.load_state_dict(checkpoint['state_dict'])
    prep = Preprocessor.restore(cfg, checkpoint['preprocessing'])
    df['Predicted Bonded Thickness (nm)'] = predict(model, prep, df)
    if args.output.exists() or args.output.resolve() == args.data.resolve():
        raise ValueError('Choose a new output file to preserve existing data.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
