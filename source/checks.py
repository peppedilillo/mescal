import source.errors as err


def check_results(calibration, data, waste, configuration):
    results = []
    if ("filter_retrigger" in configuration) and (
        not configuration["filter_retrigger"]
    ):
        results.append("filter_retrigger_off")
    if ("filter_spurious" in configuration) and (not configuration["filter_spurious"]):
        results.append("filter_spurious_off")
    if calibration.flagged:
        results.append("flagged_channels")
    if _check_time_outliers(data):
        results.append("time_outliers")
    if _check_filtered(data, waste):
        results.append("too_many_filtered_events")
    return results


def _check_time_outliers(data):
    if len(data) <= 0:
        raise err.BadDataError("Empty data.")

    mask = (data["TIME"] > 3 * data["TIME"].quantile(0.99)) | (data["TIME"] < 0)
    if data[mask].empty:
        return False
    return True


def _check_filtered(data, waste, threshold=0.50):
    if len(data) <= 0:
        raise err.BadDataError("Empty data.")

    if len(waste) / len(data) < threshold:
        return False
    return True
