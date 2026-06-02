# ===========================================================================
# Entry point (RENDER FIXED VERSION)
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Web Dashboard for Object Detection & Tracking"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 5000))
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    state.source = args.source
    state.model_name = args.model
    state.conf_threshold = args.conf
    state.device = args.device

    print("=" * 60)
    print("Object Detection Dashboard Starting")
    print("=" * 60)

    print("[INFO] Pre-loading model...")
    state.build_components()
    print("[INFO] Ready.")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", args.port)),
        debug=False,
        threaded=True
    )
