"""
DISTIX - Coordinate System Manager
Gestione dei sistemi di riferimento geodetici e cartesiani locali (ENU / UTM).
Garantisce la stabilità numerica sottraendo un'origine locale di riferimento.
"""

from typing import Tuple, Optional, Union
import numpy as np

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


class CoordinateManager:
    """
    Gestisce la conversione tra coordinate geografiche (WGS84 Lat, Lon, Alt)
    e coordinate metriche cartesiane (UTM o ENU locale).
    Mantiene un'origine locale (X0, Y0, Z0) per evitare la perdita di precisione
    a virgola mobile su valori UTM elevati (es. Y ~ 7.400.000 m).
    """

    def __init__(self, origin_xyz: Optional[np.ndarray] = None, utm_zone: Optional[int] = None, is_northern: bool = True):
        self.origin_xyz = np.zeros(3, dtype=np.float64)
        if origin_xyz is not None:
            self.origin_xyz = np.asarray(origin_xyz, dtype=np.float64).copy()
        
        self.utm_zone = utm_zone
        self.is_northern = is_northern
        self.transformer_to_utm = None
        self.transformer_to_wgs84 = None
        self.frame_mode: str = "UTM"
        self.enu_origin_wgs84: Optional[Tuple[float, float, float]] = None
        self._ecef_origin: Optional[np.ndarray] = None
        self._R_ecef_to_enu: Optional[np.ndarray] = None
        
        if HAS_PYPROJ and utm_zone is not None:
            epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
            self.transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            self.transformer_to_wgs84 = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    @staticmethod
    def _wgs84_to_ecef(lon_deg: float, lat_deg: float, alt_m: float) -> np.ndarray:
        """Conversione geodetica esatta WGS84 (lon, lat, alt) -> ECEF (X, Y, Z) in metri."""
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = f * (2.0 - f)
        lat_rad = np.deg2rad(float(lat_deg))
        lon_rad = np.deg2rad(float(lon_deg))
        sin_lat = float(np.sin(lat_rad))
        cos_lat = float(np.cos(lat_rad))
        n_val = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
        x = (n_val + float(alt_m)) * cos_lat * float(np.cos(lon_rad))
        y = (n_val + float(alt_m)) * cos_lat * float(np.sin(lon_rad))
        z = (n_val * (1.0 - e2) + float(alt_m)) * sin_lat
        return np.array([x, y, z], dtype=np.float64)

    def set_local_enu_origin(self, lon_deg: float, lat_deg: float, alt_m: float):
        """
        Configura il sistema cartesiano tangente ENU locale centrato su (lon_deg, lat_deg, alt_m).
        Allinea +Y al Nord Vero (geografico) e +X all'Est Vero eliminando la convergenza dei meridiani UTM.
        """
        self.enu_origin_wgs84 = (float(lon_deg), float(lat_deg), float(alt_m))
        self._ecef_origin = self._wgs84_to_ecef(lon_deg, lat_deg, alt_m)
        lat_r = np.deg2rad(float(lat_deg))
        lon_r = np.deg2rad(float(lon_deg))
        s_lat, c_lat = float(np.sin(lat_r)), float(np.cos(lat_r))
        s_lon, c_lon = float(np.sin(lon_r)), float(np.cos(lon_r))
        self._R_ecef_to_enu = np.array([
            [-s_lon,          c_lon,         0.0],
            [-s_lat * c_lon, -s_lat * s_lon, c_lat],
            [ c_lat * c_lon,  c_lat * s_lon, s_lat],
        ], dtype=np.float64)
        self.frame_mode = "ENU_LOCAL"

    def wgs84_to_local(self, lon: float, lat: float, alt: float) -> np.ndarray:
        """
        Converte WGS84 (lon, lat, alt) in coordinate metriche cartesiane locali:
        - Se è impostata un'origine ENU locale, restituisce ENU tangente allineato al Nord Vero (compatibile con IMU DJI).
        - Altrimenti proietta in UTM e sottrae origin_xyz (retrocompatibile).
        """
        if self._ecef_origin is not None and self._R_ecef_to_enu is not None:
            ecef = self._wgs84_to_ecef(lon, lat, alt)
            return self._R_ecef_to_enu @ (ecef - self._ecef_origin)
        utm_xyz = np.array(self.wgs84_to_utm(lon, lat, alt), dtype=np.float64)
        return self.to_local(utm_xyz)

    def set_origin(self, origin_xyz: np.ndarray):
        """Imposta l'origine cartesiana locale."""
        self.origin_xyz = np.asarray(origin_xyz, dtype=np.float64).copy()

    def to_local(self, world_xyz: np.ndarray) -> np.ndarray:
        """Da coordinate assolute (es. UTM) a coordinate relative centrate nell'origine locale."""
        arr = np.asarray(world_xyz, dtype=np.float64)
        return arr - self.origin_xyz

    def to_world(self, local_xyz: np.ndarray) -> np.ndarray:
        """Da coordinate relative locali a coordinate metriche assolute."""
        arr = np.asarray(local_xyz, dtype=np.float64)
        return arr + self.origin_xyz

    def wgs84_to_utm(self, lon: float, lat: float, alt: float) -> Tuple[float, float, float]:
        """Converte coordinate WGS84 geografiche in coordinate metriche UTM."""
        if not HAS_PYPROJ or self.transformer_to_utm is None:
            # Stima locale approssimata se pyproj non è configurato con zona esplicita
            # 1 deg lat ~ 111132 m, 1 deg lon ~ 111132 * cos(lat) m
            lat_rad = np.deg2rad(lat)
            x = lon * 111320.0 * np.cos(lat_rad)
            y = lat * 110540.0
            return float(x), float(y), float(alt)
        
        easting, northing = self.transformer_to_utm.transform(lon, lat)
        return float(easting), float(northing), float(alt)

    def utm_to_wgs84(self, easting: float, northing: float, alt: float) -> Tuple[float, float, float]:
        """Converte coordinate metriche UTM in coordinate WGS84 geografiche."""
        if not HAS_PYPROJ or self.transformer_to_wgs84 is None:
            raise RuntimeError("Pyproj non disponibile o zona UTM non configurata.")
        lon, lat = self.transformer_to_wgs84.transform(easting, northing)
        return float(lon), float(lat), float(alt)
