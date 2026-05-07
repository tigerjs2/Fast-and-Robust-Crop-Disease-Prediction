import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import torch


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "TextEmbeddings"


def load_embedding_tensor(pt_path: Path) -> torch.Tensor:
	"""Load a saved CLIP embedding tensor from a .pt file."""
	if not pt_path.exists():
		raise FileNotFoundError(f"Embedding file not found: {pt_path}")

	obj = torch.load(str(pt_path), map_location="cpu")
	if not isinstance(obj, torch.Tensor):
		raise TypeError(f"Expected a torch.Tensor in {pt_path}, got {type(obj).__name__}")

	tensor = obj.detach().cpu().float()
	if tensor.dim() == 1:
		tensor = tensor.unsqueeze(0)
	if tensor.dim() != 2:
		raise ValueError(
			f"Expected a 1D or 2D tensor in {pt_path}, got shape {tuple(tensor.shape)}"
		)
	return tensor


def tensor_to_csv_rows(tensor: torch.Tensor) -> List[List[float]]:
	"""Convert a 2D tensor into CSV rows with one embedding per row."""
	return tensor.tolist()


def write_tensor_csv(pt_path: Path, csv_path: Path) -> None:
	"""Write one .pt embedding file to a .csv file."""
	tensor = load_embedding_tensor(pt_path)
	rows = tensor_to_csv_rows(tensor)

	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		header = [f"dim_{idx}" for idx in range(tensor.size(1))]
		writer.writerow(["embedding_index", *header])
		for row_idx, row in enumerate(rows):
			writer.writerow([row_idx, *row])


def iter_pt_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
	pattern = "**/*.pt" if recursive else "*.pt"
	yield from sorted(input_dir.glob(pattern))


def convert_directory(input_dir: Path, output_dir: Path, recursive: bool) -> List[Path]:
	"""Convert every .pt file in a directory tree into a mirrored .csv tree."""
	if not input_dir.exists() or not input_dir.is_dir():
		raise NotADirectoryError(f"Invalid input directory: {input_dir}")

	written: List[Path] = []
	for pt_path in iter_pt_files(input_dir, recursive=recursive):
		rel_path = pt_path.relative_to(input_dir)
		csv_path = output_dir / rel_path.with_suffix(".csv")
		write_tensor_csv(pt_path, csv_path)
		written.append(csv_path)
	return written


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Convert CLIP text embedding .pt files to CSV")
	parser.add_argument(
		"--input_dir",
		type=str,
		default=str(DEFAULT_INPUT_DIR),
		help="Directory containing .pt embedding files",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Directory to save .csv files (default: same directory as input)",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Search for .pt files recursively under input_dir",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir) if args.output_dir else input_dir

	written = convert_directory(input_dir=input_dir, output_dir=output_dir, recursive=args.recursive)
	print(f"Converted {len(written)} file(s)")
	for path in written:
		print(path)


if __name__ == "__main__":
	main()
