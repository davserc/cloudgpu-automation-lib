#!/usr/bin/env python3
"""
Vast.ai GPU CLI

Command-line interface for managing GPU instances on Vast.ai
"""

import argparse
import json
import sys
from typing import Optional

from vast_gpu_manager import VastGPUManager, DOCKER_IMAGES


def get_manager(api_key: Optional[str] = None) -> VastGPUManager:
    """Get manager instance, with error handling"""
    try:
        return VastGPUManager(api_key=api_key)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_search(args):
    """Search for available GPUs"""
    manager = get_manager(args.api_key)

    results = manager.search_gpus(
        gpu_name=args.gpu,
        gpu_family=args.gpu_family,
        min_model=args.min_model,
        num_gpus=args.num_gpus,
        min_gpu_ram=args.min_gpu_ram,
        min_cpu_ram=args.min_cpu_ram,
        max_price=args.max_price,
        min_reliability=args.min_reliability,
        min_cuda=args.min_cuda,
        min_dlperf=args.min_dlperf,
        min_tflops=args.min_tflops,
        limit=args.limit,
        use_defaults=args.use_defaults,
        order_by=args.order_by,
        order_desc=args.desc,
    )

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    if not isinstance(results, list) or len(results) == 0:
        print("No offers found matching your criteria")
        return

    print(f"Found {len(results)} offers:\n")

    # Header
    print(
        f"{'ID':<10} "
        f"{'GPU':<16} "
        f"{'#':<2} "
        f"{'VRAM':<5} "
        f"{'$/hr':<7} "
        f"{'DLP':<6} "
        f"{'TF':<5} "
        f"{'DLP/$':<6} "
        f"{'CUDA':<5} "
        f"{'Rel':<4} "
        f"{'Location':<10}"
    )
    print("-" * 90)

    last_price_tier = None
    for offer in results:
        # Convert VRAM from MB to GB for display
        vram_gb = offer.get('gpu_ram', 0) / 1024

        # Calculate value: DLPerf per dollar (higher is better)
        price = offer.get('dph_total', 0)
        dlperf = offer.get('dlperf', 0)
        value = dlperf / price if price > 0 else 0

        # Group by price tier (round to nearest cent)
        price_tier = round(price, 2)
        if last_price_tier is not None and price_tier != last_price_tier:
            print()  # Blank line between price tiers
        last_price_tier = price_tier

        # Get location (truncate if too long)
        location = offer.get('geolocation', 'N/A')
        if len(location) > 10:
            location = location[:8] + ".."

        print(
            f"{offer.get('id', 'N/A'):<10} "
            f"{offer.get('gpu_name', 'N/A'):<16} "
            f"{offer.get('num_gpus', 0):<2} "
            f"{vram_gb:<4.0f}G "
            f"${price:<6.3f} "
            f"{dlperf:<6.1f} "
            f"{offer.get('total_flops', 0):<5.1f} "
            f"{value:<6.0f} "
            f"{offer.get('cuda_max_good', 0):<5.1f} "
            f"{offer.get('reliability', 0)*100:<4.0f} "
            f"{location:<10}"
        )


def cmd_launch(args):
    """Launch a new GPU instance"""
    manager = get_manager(args.api_key)

    # Resolve image name if it's a shortcut
    image = DOCKER_IMAGES.get(args.image, args.image)

    # Parse environment variables
    env_vars = None
    if args.env:
        env_vars = {}
        for env in args.env:
            if "=" in env:
                key, value = env.split("=", 1)
                env_vars[key] = value

    result = manager.launch_instance(
        gpu_name=args.gpu,
        image=image,
        num_gpus=args.num_gpus,
        disk_space=args.disk,
        onstart_cmd=args.onstart,
        env_vars=env_vars,
        jupyter=args.jupyter,
        ssh=not args.no_ssh
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Instance launched: {result}")


def cmd_list(args):
    """List all instances"""
    manager = get_manager(args.api_key)
    instances = manager.list_instances()

    if args.json:
        print(json.dumps(instances, indent=2, default=str))
        return

    if not isinstance(instances, list) or len(instances) == 0:
        print("No instances found")
        return

    print(f"{'ID':<12} {'Status':<15} {'GPU':<20} {'SSH':<30}")
    print("-" * 80)

    for inst in instances:
        ssh_cmd = ""
        host = inst.get("ssh_host")
        port = inst.get("ssh_port")
        if host and port:
            ssh_cmd = f"ssh -p {port} root@{host}"

        print(
            f"{inst.get('id', 'N/A'):<12} "
            f"{inst.get('actual_status', 'unknown'):<15} "
            f"{inst.get('gpu_name', 'N/A'):<20} "
            f"{ssh_cmd:<30}"
        )


def cmd_start(args):
    """Start an instance"""
    manager = get_manager(args.api_key)
    result = manager.start_instance(args.instance_id)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Instance {args.instance_id} started")


def cmd_stop(args):
    """Stop an instance"""
    manager = get_manager(args.api_key)
    result = manager.stop_instance(args.instance_id)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Instance {args.instance_id} stopped")


def cmd_destroy(args):
    """Destroy an instance"""
    manager = get_manager(args.api_key)

    if not args.force:
        confirm = input(f"Are you sure you want to destroy instance {args.instance_id}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return

    result = manager.destroy_instance(args.instance_id)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Instance {args.instance_id} destroyed")


def cmd_ssh(args):
    """Get SSH command for an instance"""
    manager = get_manager(args.api_key)
    ssh_cmd = manager.get_ssh_command(args.instance_id)

    if ssh_cmd:
        print(ssh_cmd)
    else:
        print(f"SSH not available for instance {args.instance_id}", file=sys.stderr)
        sys.exit(1)


def cmd_images(args):
    """List available Docker image shortcuts"""
    print("Available image shortcuts:\n")
    for name, image in DOCKER_IMAGES.items():
        print(f"  {name:<15} -> {image}")
    print("\nYou can also use any Docker image directly.")


def main():
    parser = argparse.ArgumentParser(
        description="Vast.ai GPU Instance Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global options
    parser.add_argument(
        "--api-key", "-k",
        help="Vast.ai API key (default: from .env or VASTAI_API_KEY)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for available GPUs")
    search_parser.add_argument("--gpu", "-g", help="Exact GPU model (e.g., RTX_4090, A100). Omit to search all GPUs")
    search_parser.add_argument("--gpu-family", "-f", help="GPU family prefix (e.g., RTX, GTX, Tesla, A100, H100)")
    search_parser.add_argument("--min-model", type=int, help="Minimum GPU model number (e.g., 3080 = RTX 3080+)")
    search_parser.add_argument("--num-gpus", "-n", type=int, help="Minimum number of GPUs (default: 1)")
    search_parser.add_argument("--min-gpu-ram", "-r", type=float, help="Minimum GPU RAM in GB")
    search_parser.add_argument("--min-cpu-ram", type=float, help="Minimum system RAM in GB")
    search_parser.add_argument("--max-price", "-p", type=float, help="Maximum price per hour (USD). Omit for no limit")
    search_parser.add_argument("--min-reliability", type=float, help="Minimum reliability 0-1. Omit for no filter")
    search_parser.add_argument("--min-cuda", type=float, help="Minimum CUDA version (e.g., 12.0)")
    search_parser.add_argument("--min-dlperf", type=float, help="Minimum DLPerf score (deep learning performance)")
    search_parser.add_argument("--min-tflops", type=float, help="Minimum TFLOPS (compute power)")
    search_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results (default: 20)")
    search_parser.add_argument("--use-defaults", "-d", action="store_true", help="Use defaults from .env file")
    search_parser.add_argument(
        "--order-by", "-o",
        choices=["price", "gpu_ram", "reliability", "num_gpus", "score", "dlperf", "cpu_ram", "disk_space", "tflops", "cuda", "value", "tflops_value", "price_power"],
        default="price",
        help="Sort results by field (default: price). 'value' = DLPerf/$, 'price_power' = price asc + power desc"
    )
    search_parser.add_argument("--desc", action="store_true", help="Sort descending (default: ascending)")
    search_parser.set_defaults(func=cmd_search, use_defaults=False, desc=False)

    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch a new GPU instance")
    launch_parser.add_argument("--gpu", "-g", help="GPU model (e.g., RTX_4090, A100)")
    launch_parser.add_argument("--image", "-i", default="pytorch", help="Docker image or shortcut (default: pytorch)")
    launch_parser.add_argument("--num-gpus", "-n", type=int, help="Number of GPUs")
    launch_parser.add_argument("--disk", "-d", type=float, default=20.0, help="Disk space in GB (default: 20)")
    launch_parser.add_argument("--onstart", help="Command to run on start")
    launch_parser.add_argument("--env", "-e", action="append", help="Environment variable (KEY=VALUE)")
    launch_parser.add_argument("--jupyter", action="store_true", help="Enable Jupyter notebook")
    launch_parser.add_argument("--no-ssh", action="store_true", help="Disable SSH access")
    launch_parser.set_defaults(func=cmd_launch)

    # List command
    list_parser = subparsers.add_parser("list", help="List your instances")
    list_parser.set_defaults(func=cmd_list)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start a stopped instance")
    start_parser.add_argument("instance_id", type=int, help="Instance ID")
    start_parser.set_defaults(func=cmd_start)

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop a running instance")
    stop_parser.add_argument("instance_id", type=int, help="Instance ID")
    stop_parser.set_defaults(func=cmd_stop)

    # Destroy command
    destroy_parser = subparsers.add_parser("destroy", help="Destroy an instance")
    destroy_parser.add_argument("instance_id", type=int, help="Instance ID")
    destroy_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    destroy_parser.set_defaults(func=cmd_destroy)

    # SSH command
    ssh_parser = subparsers.add_parser("ssh", help="Get SSH command for an instance")
    ssh_parser.add_argument("instance_id", type=int, help="Instance ID")
    ssh_parser.set_defaults(func=cmd_ssh)

    # Images command
    images_parser = subparsers.add_parser("images", help="List Docker image shortcuts")
    images_parser.set_defaults(func=cmd_images)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
