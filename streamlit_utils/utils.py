def get_config_rows(config):
    rows = []

    for key, value in config.items():
        if key == "timestamp":
            continue
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                rows.append(
                    {
                        "Parameter": f"{key} → {subkey}",
                        "Value": str(subvalue),
                    }
                )
        else:
            rows.append(
                {
                    "Parameter": key,
                    "Value": str(value),
                }
            )
    return rows
