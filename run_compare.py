from Solve_Emix_updating import compare_edl_effects, default_params


def main() -> None:
    params = default_params()
    out = compare_edl_effects(
        params,
        save_dir="results/compare_demo",
        save_data=True,
        save_fig=True,
    )
    print(out["comparison"])


if __name__ == "__main__":
    main()
