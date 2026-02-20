#!/usr/bin/env python3
"""
Vast.ai GPU CLI

Command-line interface for managing GPU instances on Vast.ai
"""

import argparse
import json
import sys
import time

from services.vast.gpu_manager import DOCKER_IMAGES, VastGPUManager


def get_manager(api_key: str | None = None) -> VastGPUManager:
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
        vram_gb = offer.get("gpu_ram", 0) / 1024

        # Calculate value: DLPerf per dollar (higher is better)
        price = offer.get("dph_total", 0)
        dlperf = offer.get("dlperf", 0)
        value = dlperf / price if price > 0 else 0

        # Group by price tier (round to nearest cent)
        price_tier = round(price, 2)
        if last_price_tier is not None and price_tier != last_price_tier:
            print()  # Blank line between price tiers
        last_price_tier = price_tier

        # Get location (truncate if too long)
        location = offer.get("geolocation") or "N/A"
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

    # Parse offer IDs (can be comma-separated)
    offer_ids = []
    if args.offer_id:
        offer_ids = [int(x.strip()) for x in args.offer_id.split(",")]

    # Launch by offer ID(s) or by GPU name
    if offer_ids:
        results = []
        for offer_id in offer_ids:
            result = manager.launch_by_offer_id(
                offer_id=offer_id,
                image=image,
                disk_space=args.disk,
                ports=args.ports,
                onstart_cmd=args.onstart,
                env_vars=env_vars,
                jupyter=args.jupyter,
                ssh=not args.no_ssh,
            )
            results.append({"offer_id": offer_id, "result": result})

        if args.json:
            print(json.dumps(results, indent=2, default=str))
            return

        # Collect launched instance IDs
        instance_ids = []
        for r in results:
            if r["result"].get("success"):
                instance_id = r["result"].get("new_contract")
                instance_ids.append(instance_id)
                print(f"Instance {instance_id} launched (offer {r['offer_id']})")
            else:
                print(f"Failed to launch offer {r['offer_id']}: {r['result']}")

        if not instance_ids:
            return

        # Wait for SSH on all instances
        print("\nWaiting for SSH to be ready", end="", flush=True)
        ssh_commands = {}
        for _ in range(12):
            time.sleep(5)
            print(".", end="", flush=True)
            instances = manager.list_instances()
            for inst in instances:
                inst_id = inst.get("id")
                if inst_id in instance_ids and inst_id not in ssh_commands:
                    host = inst.get("ssh_host")
                    port = inst.get("ssh_port")
                    if host and port:
                        ssh_commands[inst_id] = f"ssh -p {port} root@{host}"
            if len(ssh_commands) == len(instance_ids):
                break

        print()
        if ssh_commands:
            print()
            for inst_id, ssh_cmd in ssh_commands.items():
                print(f"{inst_id}: {ssh_cmd}")
        else:
            print("\nSSH not ready yet. Check with: python cli.py list")
    else:
        result = manager.launch_instance(
            gpu_name=args.gpu,
            image=image,
            num_gpus=args.num_gpus,
            disk_space=args.disk,
            ports=args.ports,
            onstart_cmd=args.onstart,
            env_vars=env_vars,
            jupyter=args.jupyter,
            ssh=not args.no_ssh,
        )

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if result.get("success"):
                instance_id = result.get("new_contract")
                print(f"Instance {instance_id} launched successfully!")
                print("\nWaiting for SSH to be ready", end="", flush=True)

                ssh_cmd = None
                for _ in range(12):
                    time.sleep(5)
                    print(".", end="", flush=True)
                    instances = manager.list_instances()
                    for inst in instances:
                        if inst.get("id") == instance_id:
                            host = inst.get("ssh_host")
                            port = inst.get("ssh_port")
                            if host and port:
                                ssh_cmd = f"ssh -p {port} root@{host}"
                                break
                    if ssh_cmd:
                        break

                print()
                if ssh_cmd:
                    print(f"\n{ssh_cmd}")
                else:
                    print("\nSSH not ready yet. Check with: python cli.py list")
            else:
                print(f"Launch failed: {result}")


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
            f"{inst.get('id') or 'N/A':<12} "
            f"{inst.get('actual_status') or 'unknown':<15} "
            f"{inst.get('gpu_name') or 'N/A':<20} "
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
    """Destroy an instance or all instances"""
    manager = get_manager(args.api_key)

    # Destroy all instances
    if args.all:
        instances = manager.list_instances()
        if not instances:
            print("No instances to destroy")
            return

        instance_ids = [inst.get("id") for inst in instances if inst.get("id")]
        if not instance_ids:
            print("No instances to destroy")
            return

        if not args.force:
            print(f"Instances to destroy: {', '.join(map(str, instance_ids))}")
            confirm = input(
                f"Are you sure you want to destroy ALL {len(instance_ids)} instances? [y/N]: "
            )
            if confirm.lower() != "y":
                print("Cancelled")
                return

        results = []
        for inst_id in instance_ids:
            result = manager.destroy_instance(inst_id)
            results.append({"instance_id": inst_id, "result": result})
            print(f"Instance {inst_id} destroyed")

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        return

    # Destroy single instance
    if args.instance_id is None:
        print("Error: specify instance_id or use --all", file=sys.stderr)
        sys.exit(1)

    if not args.force:
        confirm = input(f"Are you sure you want to destroy instance {args.instance_id}? [y/N]: ")
        if confirm.lower() != "y":
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


def cmd_balance(args):
    """Show account balance"""
    manager = get_manager(args.api_key)
    balance = manager.get_balance()

    if args.json:
        print(json.dumps(balance, indent=2, default=str))
    else:
        print(f"Account: {balance.get('email', 'N/A')}")
        print(f"Credit:  ${balance.get('credit', 0):.2f}")
        print(f"Spent:   ${balance.get('total_spend', 0):.4f}")


def cmd_billing(args):
    """Show billing history"""
    from datetime import datetime

    manager = get_manager(args.api_key)
    invoices = manager.get_invoices(limit=args.limit)

    if args.json:
        print(json.dumps(invoices, indent=2, default=str))
        return

    if not invoices:
        print("No billing history found")
        return

    print(f"{'Date':<12} {'Type':<10} {'Amount':<10} {'Description':<50}")
    print("-" * 85)

    for inv in invoices:
        ts = inv.get("timestamp", 0)
        if ts > 1e12:  # milliseconds
            ts = ts / 1000
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "N/A"

        inv_type = inv.get("type", "N/A")
        amount = float(inv.get("amount", 0))

        # Format amount with sign
        if inv_type == "payment":
            amount_str = f"+${abs(amount):.2f}"
        else:
            amount_str = f"-${abs(amount):.3f}"

        desc = inv.get("description", "")
        if not desc and inv_type == "payment":
            desc = f"Payment ({inv.get('network', '')} {inv.get('last4', '')})"

        # Truncate description
        if len(desc) > 48:
            desc = desc[:46] + ".."

        print(f"{date_str:<12} {inv_type:<10} {amount_str:<10} {desc:<50}")


def cmd_images(args):
    """List available Docker image shortcuts"""
    print("Available image shortcuts:\n")
    for name, image in DOCKER_IMAGES.items():
        print(f"  {name:<15} -> {image}")
    print("\nYou can also use any Docker image directly.")


def cmd_ssh_key(args):
    """Upload SSH key to Vast.ai account"""
    from pathlib import Path

    key_path = Path(args.key_file).expanduser()

    if not key_path.exists():
        print(f"Error: Key file not found: {key_path}", file=sys.stderr)
        sys.exit(1)

    public_key = key_path.read_text().strip()

    if not public_key.startswith(("ssh-rsa", "ssh-ed25519", "ecdsa-", "ssh-dss")):
        print("Error: File doesn't look like a public SSH key", file=sys.stderr)
        sys.exit(1)

    manager = get_manager(args.api_key)
    try:
        result = manager.add_ssh_key(public_key)
        # SDK returns empty string or None on success
        if result is None or result == "":
            print("SSH key uploaded successfully")
        elif isinstance(result, dict) and (result.get("success") or result.get("id")):
            print("SSH key uploaded successfully")
        elif isinstance(result, str) and result:
            # Non-empty string is usually an error or message
            print(f"Result: {result}")
        else:
            print("SSH key uploaded successfully")
    except Exception as e:
        print(f"Failed to upload SSH key: {e}", file=sys.stderr)
        sys.exit(1)


def get_ssh_info(manager, instance_id: int) -> tuple[str, int] | None:
    """Get SSH host and port for an instance."""
    instance = manager.get_instance(instance_id)
    if instance:
        host = instance.get("ssh_host")
        port = instance.get("ssh_port")
        if host and port:
            return (host, port)
    return None


def cmd_upload(args):
    """Upload files to an instance via SCP."""
    import subprocess

    manager = get_manager(args.api_key)
    ssh_info = get_ssh_info(manager, args.instance_id)

    if not ssh_info:
        print(f"SSH not available for instance {args.instance_id}", file=sys.stderr)
        sys.exit(1)

    host, port = ssh_info
    dst = args.dst or "/root/"

    # Build scp command
    scp_cmd = [
        "scp",
        "-P",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-r",  # recursive
        args.src,
        f"root@{host}:{dst}",
    ]

    print(f"Uploading {args.src} -> {host}:{dst}")
    result = subprocess.run(scp_cmd)
    sys.exit(result.returncode)


def cmd_download(args):
    """Download files from an instance via SCP."""
    import subprocess

    manager = get_manager(args.api_key)
    ssh_info = get_ssh_info(manager, args.instance_id)

    if not ssh_info:
        print(f"SSH not available for instance {args.instance_id}", file=sys.stderr)
        sys.exit(1)

    host, port = ssh_info
    dst = args.dst or "./"

    # Build scp command
    scp_cmd = [
        "scp",
        "-P",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-r",  # recursive
        f"root@{host}:{args.src}",
        dst,
    ]

    print(f"Downloading {host}:{args.src} -> {dst}")
    result = subprocess.run(scp_cmd)
    sys.exit(result.returncode)


def cmd_run(args):
    """Execute a command on an instance via SSH."""
    import subprocess

    manager = get_manager(args.api_key)
    ssh_info = get_ssh_info(manager, args.instance_id)

    if not ssh_info:
        print(f"SSH not available for instance {args.instance_id}", file=sys.stderr)
        sys.exit(1)

    host, port = ssh_info

    # Build ssh command
    ssh_cmd = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        f"root@{host}",
        args.cmd,
    ]

    result = subprocess.run(ssh_cmd)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Vast.ai GPU Instance Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--api-key", "-k", help="Vast.ai API key (default: from .env or VASTAI_API_KEY)"
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output in JSON format")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for available GPUs")
    search_parser.add_argument(
        "--gpu", "-g", help="Exact GPU model (e.g., RTX_4090, A100). Omit to search all GPUs"
    )
    search_parser.add_argument(
        "--gpu-family", "-f", help="GPU family prefix (e.g., RTX, GTX, Tesla, A100, H100)"
    )
    search_parser.add_argument(
        "--min-model", type=int, help="Minimum GPU model number (e.g., 3080 = RTX 3080+)"
    )
    search_parser.add_argument(
        "--num-gpus", "-n", type=int, help="Minimum number of GPUs (default: 1)"
    )
    search_parser.add_argument("--min-gpu-ram", "-r", type=float, help="Minimum GPU RAM in GB")
    search_parser.add_argument("--min-cpu-ram", type=float, help="Minimum system RAM in GB")
    search_parser.add_argument(
        "--max-price", "-p", type=float, help="Maximum price per hour (USD). Omit for no limit"
    )
    search_parser.add_argument(
        "--min-reliability", type=float, help="Minimum reliability 0-1. Omit for no filter"
    )
    search_parser.add_argument("--min-cuda", type=float, help="Minimum CUDA version (e.g., 12.0)")
    search_parser.add_argument(
        "--min-dlperf", type=float, help="Minimum DLPerf score (deep learning performance)"
    )
    search_parser.add_argument("--min-tflops", type=float, help="Minimum TFLOPS (compute power)")
    search_parser.add_argument(
        "--limit", "-l", type=int, default=20, help="Max results (default: 20)"
    )
    search_parser.add_argument(
        "--use-defaults", "-d", action="store_true", help="Use defaults from .env file"
    )
    search_parser.add_argument(
        "--order-by",
        "-o",
        choices=[
            "price",
            "gpu_ram",
            "reliability",
            "num_gpus",
            "score",
            "dlperf",
            "cpu_ram",
            "disk_space",
            "tflops",
            "cuda",
            "value",
            "tflops_value",
            "price_power",
        ],
        default="price",
        help="Sort results by field (default: price). 'value' = DLPerf/$, 'price_power' = price asc + power desc",
    )
    search_parser.add_argument(
        "--desc", action="store_true", help="Sort descending (default: ascending)"
    )
    search_parser.set_defaults(func=cmd_search, use_defaults=False, desc=False)

    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch a new GPU instance")
    launch_parser.add_argument(
        "--offer-id", "--id", help="Offer ID(s) from search (comma-separated for multiple)"
    )
    launch_parser.add_argument(
        "--gpu", "-g", help="GPU model (e.g., RTX_4090, A100). Ignored if --offer-id is set"
    )
    launch_parser.add_argument(
        "--image", "-i", default="pytorch", help="Docker image or shortcut (default: pytorch)"
    )
    launch_parser.add_argument("--num-gpus", "-n", type=int, help="Number of GPUs")
    launch_parser.add_argument(
        "--disk", "-d", type=float, default=20.0, help="Disk space in GB (default: 20)"
    )
    launch_parser.add_argument(
        "--ports",
        help="Ports to expose (comma-separated, e.g., 5000,8888)",
    )
    launch_parser.add_argument("--onstart", help="Command to run on start")
    launch_parser.add_argument(
        "--env", "-e", action="append", help="Environment variable (KEY=VALUE)"
    )
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
    destroy_parser = subparsers.add_parser("destroy", help="Destroy an instance or all instances")
    destroy_parser.add_argument(
        "instance_id", type=int, nargs="?", help="Instance ID (optional if using --all)"
    )
    destroy_parser.add_argument("--all", "-a", action="store_true", help="Destroy ALL instances")
    destroy_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    destroy_parser.set_defaults(func=cmd_destroy)

    # SSH command
    ssh_parser = subparsers.add_parser("ssh", help="Get SSH command for an instance")
    ssh_parser.add_argument("instance_id", type=int, help="Instance ID")
    ssh_parser.set_defaults(func=cmd_ssh)

    # Images command
    images_parser = subparsers.add_parser("images", help="List Docker image shortcuts")
    images_parser.set_defaults(func=cmd_images)

    # Balance command
    balance_parser = subparsers.add_parser("balance", help="Show account balance")
    balance_parser.set_defaults(func=cmd_balance)

    # Billing command
    billing_parser = subparsers.add_parser("billing", help="Show billing history")
    billing_parser.add_argument(
        "--limit", "-l", type=int, default=20, help="Max transactions (default: 20)"
    )
    billing_parser.set_defaults(func=cmd_billing)

    # SSH key command
    ssh_key_parser = subparsers.add_parser("ssh-key", help="Upload SSH key to Vast.ai")
    ssh_key_parser.add_argument(
        "key_file",
        nargs="?",
        default="~/.ssh/id_ed25519.pub",
        help="Path to public key file (default: ~/.ssh/id_ed25519.pub)",
    )
    ssh_key_parser.set_defaults(func=cmd_ssh_key)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload files to instance via SCP")
    upload_parser.add_argument("instance_id", type=int, help="Instance ID")
    upload_parser.add_argument("src", help="Local source path (file or directory)")
    upload_parser.add_argument("--dst", "-d", help="Remote destination path (default: /root/)")
    upload_parser.set_defaults(func=cmd_upload)

    # Download command
    download_parser = subparsers.add_parser("download", help="Download files from instance via SCP")
    download_parser.add_argument("instance_id", type=int, help="Instance ID")
    download_parser.add_argument("src", help="Remote source path (file or directory)")
    download_parser.add_argument("--dst", "-d", help="Local destination path (default: ./)")
    download_parser.set_defaults(func=cmd_download)

    # Run command
    run_parser = subparsers.add_parser("run", help="Execute command on instance via SSH")
    run_parser.add_argument("instance_id", type=int, help="Instance ID")
    run_parser.add_argument("cmd", help="Command to execute")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
