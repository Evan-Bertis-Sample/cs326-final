from analysis.config import AnalysisConfig
import pandas as pd
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Dict, List, Set, Tuple, Iterable
import numpy as np


@dataclass(frozen=True, slots=True)
class GeoID:
    country: str
    region: Optional[str] = None

    def __str__(self) -> str:
        tpl = getattr(AnalysisConfig.metadata, "geo_id_template", "{country}.{region}")
        s = tpl.format(country=self.country or "", region=(self.region or ""))
        return s.strip(".").strip()

    @staticmethod
    def keys() -> Dict[str, str]:
        md = AnalysisConfig.metadata
        return {
            "country_code": getattr(md, "country_code_key", "CountryCode"),
            "country_name": getattr(md, "country_name_key", "CountryName"),
            "region_name": getattr(md, "region_name_key", "RegionName"),
            "region_code": getattr(md, "region_code_key", "RegionCode"),
        }

    @classmethod
    def choose_country_series(cls, df: pd.DataFrame) -> pd.Series:
        k = cls.keys()
        return df[k["country_code"]] if k["country_code"] in df.columns else df[k["country_name"]]

    @classmethod
    def choose_region_series(cls, df: pd.DataFrame) -> Optional[pd.Series]:
        k = cls.keys()
        if k["region_name"] in df.columns:
            return df[k["region_name"]]
        if k["region_code"] in df.columns:
            return df[k["region_code"]]
        return None

    @staticmethod
    def _dedupe_order_preserving(items: Iterable["GeoID"]) -> List["GeoID"]:
        seen: set[str] = set()
        out: List[GeoID] = []
        for g in items:
            key = str(g)
            if key and key not in seen:
                seen.add(key)
                out.append(g)
        return out

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, unique: bool = False) -> List["GeoID"]:
        c = cls.choose_country_series(df).astype("string").str.strip()
        r = cls.choose_region_series(df)
        if r is None:
            objs = [cls(country=(str(cc) if pd.notna(cc) else "")) for cc in c]
        else:
            r_str = r.astype("string").str.strip()
            objs = [
                cls(
                    country=(str(cc) if pd.notna(cc) else ""),
                    region=(str(rr) if pd.notna(rr) and str(rr).strip() != "" else None),
                )
                for cc, rr in zip(c, r_str)
            ]
        return cls._dedupe_order_preserving(objs) if unique else objs

    @classmethod
    def from_strings(cls, names: Iterable[str], unique: bool = False) -> List["GeoID"]:
        out: List[GeoID] = []
        for s in names:
            s = s.strip()
            if "." in s:
                a, b = s.split(".", 1)
                out.append(cls(country=a.strip(), region=b.strip() or None))
            else:
                out.append(cls(country=s))
        return cls._dedupe_order_preserving(out) if unique else out

    @staticmethod
    def to_strings(items: Iterable["GeoID"]) -> List[str]:
        return [str(g) for g in items]

    @staticmethod
    def to_string_series(items: Iterable["GeoID"], index: Optional[pd.Index] = None) -> pd.Series:
        vals = [str(g) for g in items]
        return pd.Series(vals, index=index, dtype="string")

    @staticmethod
    def to_string_set(items: Iterable["GeoID"]) -> Set[str]:
        return set(str(g) for g in items if str(g))

class DataTuple(tp.NamedTuple):
    training: pd.DataFrame
    testing: pd.DataFrame
    validation: Optional[pd.DataFrame] = None


class OxCGRTData:
    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, (str, Path)):
            self.data = pd.read_csv(data, low_memory=False)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise TypeError("data must be a CSV path or a pandas DataFrame")

        self._ensure_required_columns()
        self._attach_geoid_column()

    @staticmethod
    def required_columns() -> Set[str]:
        md = AnalysisConfig.metadata
        return set(md.id_columns + md.policy_columns + md.outcome_columns)

    def _ensure_required_columns(self) -> None:
        missing = self.required_columns() - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    @property
    def date_col(self) -> str:
        return AnalysisConfig.metadata.date_column

    @property
    def geoid_col(self) -> str:
        return getattr(AnalysisConfig.metadata, "geo_id_column", "GeoID")

    def _attach_geoid_column(self) -> None:
        series = GeoID.to_string_series(
            GeoID.from_dataframe(self.data, unique=False),
            index=self.data.index,
        )
        self.data[self.geoid_col] = series

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        dc = self.date_col
        if dc in df.columns and not pd.api.types.is_datetime64_any_dtype(df[dc]):
            df = df.copy()
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
        return df

    def geo_ids(self, unique: bool = False) -> List[GeoID]:
        if not unique:
            return GeoID.from_strings(self.data[self.geoid_col].tolist(), unique=False)
        uniq_strings = pd.Series(self.data[self.geoid_col], dtype="string").dropna().drop_duplicates().tolist()
        return GeoID.from_strings(uniq_strings, unique=True)

    def geo_id_strings(self, unique: bool = False) -> pd.Series:
        col = self.data[self.geoid_col].astype("string")
        if not unique:
            return col
        s = col.dropna().drop_duplicates().reset_index(drop=True)
        return s

    def filter(
        self,
        *,
        geo_ids: Optional[Iterable[str]] = None,
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        query: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> "OxCGRTData":
        df = self.data
        if geo_ids is not None:
            targets = GeoID.from_strings(geo_ids, unique=True)
            target_set = set(GeoID.to_strings(targets))
            df = df[df[self.geoid_col].isin(target_set)]
        if date_range is not None:
            df = self._ensure_datetime(df)
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df[self.date_col] >= start) & (df[self.date_col] <= end)]
        if query:
            df = df.query(query)
        if columns:
            df = df[list(columns)]
        return OxCGRTData(df)

    def get_timeseries(self, geo_id: str) -> pd.DataFrame:
        target = GeoID.from_strings([geo_id], unique=True)[0]
        df = self.data.loc[self.data[self.geoid_col] == str(target)]
        df = self._ensure_datetime(df)
        return df.sort_values(self.date_col)

    def cluster_regions(self, region_clusters: List[List[str]]) -> Dict[frozenset, pd.DataFrame]:
        out: Dict[frozenset, pd.DataFrame] = {}
        s = self.data[self.geoid_col].astype("string")
        for cluster in region_clusters:
            targets = GeoID.from_strings(cluster, unique=True)
            keys = frozenset(GeoID.to_strings(targets))
            out[keys] = self.data.loc[s.isin(keys)].copy()
        return out

    def split_by_region(
        self,
        ratio: Union[float, Tuple[float, float, float]] = (0.7, 0.15, 0.15),
        seed: int = 1,
        group_by: str = "geo",  # "geo" or "country"
    ) -> DataTuple:
        if isinstance(ratio, float):
            a, b, c = ratio, 0.0, 1.0 - ratio
        else:
            a, b, c = ratio
            if abs((a + b + c) - 1.0) > 1e-8:
                raise ValueError("ratios must sum to 1")

        if group_by == "geo":
            keys_series = self.data[self.geoid_col].astype("string")
        elif group_by == "country":
            cc_key = getattr(AnalysisConfig.metadata, "country_code_key", "CountryCode")
            cn_key = getattr(AnalysisConfig.metadata, "country_name_key", "CountryName")
            col = cc_key if cc_key in self.data.columns else cn_key
            keys_series = pd.Series(self.data[col], index=self.data.index).astype("string").str.strip()
        else:
            raise ValueError("group_by must be 'geo' or 'country'")

        uniq = pd.unique(keys_series.dropna())
        rs = np.random.RandomState(seed)
        perm = uniq.copy()
        rs.shuffle(perm)

        n = len(perm)
        na = int(round(a * n))
        nb = int(round(b * n))
        train_keys = set(perm[:na])
        val_keys = set(perm[na:na + nb])
        test_keys = set(perm[na + nb:])

        df = self.data
        train_df = df[keys_series.isin(train_keys)].copy()
        val_df = df[keys_series.isin(val_keys)].copy() if nb > 0 else None
        test_df = df[keys_series.isin(test_keys)].copy()
        return DataTuple(training=train_df, testing=test_df, validation=val_df)

    def split_rows(
        self,
        ratio: Union[float, Tuple[float, float, float]] = (0.7, 0.15, 0.15),
        seed: int = 1,
        shuffle: bool = True,
    ) -> DataTuple:
        df = self.data
        idx = df.index.to_numpy()
        if shuffle:
            rs = np.random.RandomState(seed)
            rs.shuffle(idx)
        if isinstance(ratio, float):
            cut = int(round(ratio * len(idx)))
            train_idx, test_idx, val_idx = idx[:cut], idx[cut:], idx[0:0]
        else:
            a, b, c = ratio
            if abs((a + b + c) - 1.0) > 1e-8:
                raise ValueError("ratios must sum to 1")
            n = len(idx)
            na, nb = int(round(a * n)), int(round(b * n))
            train_idx, val_idx, test_idx = idx[:na], idx[na:na + nb], idx[na + nb:]
        train_df = df.loc[train_idx].copy()
        val_df = df.loc[val_idx].copy() if len(val_idx) else None
        test_df = df.loc[test_idx].copy()
        return DataTuple(training=train_df, testing=test_df, validation=val_df)