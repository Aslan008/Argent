"""How much of the weekly Kaggle accelerator quota is left (the numbers the website shows under your profile).

    py -3.14 scripts/kaggle_quota.py          # the interpreter that has the `kaggle` package (not the project's .venv)

GPU: 30 h a week; sessions that are running RESERVE their remaining maximum, so the free quota is  allowed - used - reserved.
The window resets at the time printed (Saturday 00:00 UTC).
"""

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest


def hours(td) -> float:
    return td.total_seconds() / 3600.0


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        r = client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
    print(f"window resets {r.quota_refresh_time:%Y-%m-%d %H:%M} UTC")
    for name in ("gpu_quota", "tpu_quota"):
        q = getattr(r, name)
        if q is None:
            continue
        free = hours(q.total_time_allowed) - hours(q.time_used) - hours(q.time_reserved)
        print(f"{name[:3].upper()}: used {hours(q.time_used):5.1f} h   reserved by running sessions {hours(q.time_reserved):5.1f} h   allowed {hours(q.total_time_allowed):5.1f} h   "
              f"=> free now {free:5.1f} h")


if __name__ == "__main__":
    main()
