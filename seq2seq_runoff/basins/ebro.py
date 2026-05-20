"""Cuenca del Ebro hasta el aforo A284 (Tudela).

Contiene todo lo específico del Ebro:
    - los 9 pluviómetros, los 3 embalses y el aforo,
    - las cabeceras concretas con que vienen los CSV del operador,
    - la topología fluvial canónica derivada de los comentarios del
      notebook `Modelo-V0.0.ipynb`.

Puede servir de plantilla para escribir otras cuencas.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..basin import BasinSpec, StationSpec
from ..gnn.graph import BasinGraph


# Cabeceras de fecha con espacios al final tal como vienen del operador.
_FECHA_PRECIP = "fecha     "
_FECHA_EMBALSE = "fecha           "
_FECHA_CAUDAL = "fecha         "
_VALOR_ACUM = "Acumulado"
_VALOR_MEDIA = "Media           "


# --- BasinSpec --------------------------------------------------------------


def ebro_basin() -> BasinSpec:
    """Devuelve el `BasinSpec` de la cuenca del Ebro hasta Tudela."""
    rainfall = [
        StationSpec("EM01-PACUM", "EM01L84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
        StationSpec("EM06-PACUM", "EM06L84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
        StationSpec("EM09-PACUM", "EM09L84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
        StationSpec("EM11-PACUM", "EM11L84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
        StationSpec("EM25-PACUM", "EM25T84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
        StationSpec("EM29-PACUM", "EM29Y84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
        StationSpec("EM30-PACUM", "EM30T84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
        StationSpec("EM71-PACUM", "EM71T84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
        StationSpec("EM75-PACUM", "EM75E84PACUM", _VALOR_ACUM, _FECHA_PRECIP, "rain"),
    ]
    reservoirs = [
        StationSpec("E001", "E001L65VEMBA", _VALOR_ACUM, _FECHA_EMBALSE, "reservoir"),
        StationSpec("E029", "E029Y65VEMBA", _VALOR_ACUM, _FECHA_EMBALSE, "reservoir"),
        StationSpec("E075", "E075E65VEMBA", _VALOR_ACUM, _FECHA_EMBALSE, "reservoir"),
    ]
    flow = StationSpec("A284", "A284Z65QRIO1", _VALOR_MEDIA, _FECHA_CAUDAL, "flow")

    return BasinSpec(
        name="Ebro",
        rainfall_stations=rainfall,
        reservoirs=reservoirs,
        flow_station=flow,
        caudal_minimo_m3s=30.0,
        fecha_corte_inicial="2014-06-01",
    )


# --- BasinGraph -------------------------------------------------------------


def _make_edge_index(
    pares: List[Tuple[str, str]],
    src_idx: Dict[str, int],
    dst_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    s = np.array([src_idx[a] for a, _ in pares], dtype=np.int64)
    d = np.array([dst_idx[b] for _, b in pares], dtype=np.int64)
    return s, d


def ebro_graph() -> BasinGraph:
    """Topología canónica del Ebro hasta Tudela.

    Conectividad derivada de los comentarios del notebook original
    `Modelo-V0.0.ipynb`. Las longitudes fluviales por arista y las
    coordenadas (lat, lon) por nodo son estimaciones del operador,
    documentadas como ``confianza media/baja'' en el CSV de provenance
    (cf. §1.5 del paper). Sirven para inicializar de forma informada los
    routings $\\lambda$ y para los layouts geográficos de las figuras;
    el modelo sigue siendo libre de ajustarlos por gradiente.
    """
    type1_names: List[str] = [
        # Sub-cuencas con estación pluviométrica
        "EM01_HEAD", "EM06", "EM09", "EM11",
        "EM25", "EM29_HEAD", "EM30", "EM71", "EM75_HEAD",
        # Confluencias
        "MIRANDA", "LOGRONO", "ARGA_OUT",
        "IRATI_OUT", "ARAGON_OUT", "MEDIO_OUT",
        # Aforo
        "TUDELA",
    ]
    name2idx = {n: i for i, n in enumerate(type1_names)}

    # Tabla de arcos E_11 con longitud fluvial (km). Las longitudes son
    # estimaciones del operador (confianza media-baja); para producción
    # convendría recalcularlas con la capa hidrográfica IGN/CEDEX/CHE.
    edges_11_with_len = [
        ("EM01_HEAD", "MIRANDA", 165.0),
        ("EM09",      "MIRANDA",  70.0),
        ("MIRANDA",   "LOGRONO",  72.0),
        ("EM06",      "LOGRONO",  95.0),
        ("EM11",      "LOGRONO",  62.0),
        ("LOGRONO",   "MEDIO_OUT", 96.0),
        ("EM25",      "ARGA_OUT", 82.0),
        ("EM30",      "ARGA_OUT", 70.0),
        ("ARGA_OUT",  "MEDIO_OUT", 15.0),
        ("EM75_HEAD", "IRATI_OUT", 88.0),
        ("IRATI_OUT", "ARAGON_OUT", 70.0),
        ("EM29_HEAD", "ARAGON_OUT", 195.0),
        ("ARAGON_OUT", "MEDIO_OUT", 3.0),
        ("EM71",      "MEDIO_OUT", 35.0),
        ("MEDIO_OUT", "TUDELA",    24.0),
    ]
    edges_11 = [(a, b) for a, b, _ in edges_11_with_len]
    edge_len_km_11 = np.array([L for _, _, L in edges_11_with_len], dtype=np.float32)
    s11, d11 = _make_edge_index(edges_11, name2idx, name2idx)

    res_names = ["E001", "E029", "E075"]
    res2idx = {r: i for i, r in enumerate(res_names)}

    edges_12_with_len = [
        ("EM01_HEAD", "E001", 22.0),
        ("EM29_HEAD", "E029", 105.0),
        ("EM75_HEAD", "E075", 55.0),
    ]
    edges_12 = [(a, b) for a, b, _ in edges_12_with_len]
    len_12 = np.array([L for _, _, L in edges_12_with_len], dtype=np.float32)
    src12, dst12 = _make_edge_index(edges_12, name2idx, res2idx)

    edges_21_with_len = [
        ("E001", "MIRANDA",    143.0),
        ("E029", "ARAGON_OUT",  90.0),
        ("E075", "IRATI_OUT",   33.0),
    ]
    edges_21 = [(a, b) for a, b, _ in edges_21_with_len]
    len_21 = np.array([L for _, _, L in edges_21_with_len], dtype=np.float32)
    src21, dst21 = _make_edge_index(edges_21, res2idx, name2idx)

    rain_to_type1 = {
        "EM01-PACUM": name2idx["EM01_HEAD"],
        "EM06-PACUM": name2idx["EM06"],
        "EM09-PACUM": name2idx["EM09"],
        "EM11-PACUM": name2idx["EM11"],
        "EM25-PACUM": name2idx["EM25"],
        "EM29-PACUM": name2idx["EM29_HEAD"],
        "EM30-PACUM": name2idx["EM30"],
        "EM71-PACUM": name2idx["EM71"],
        "EM75-PACUM": name2idx["EM75_HEAD"],
    }

    # Coordenadas aproximadas (lat, lon) — confianza baja, sólo para
    # visualización geográfica de figuras.
    type1_latlon = {
        "EM01_HEAD":  (43.038, -4.190),
        "EM09":       (42.850, -2.700),
        "MIRANDA":    (42.686, -2.947),
        "LOGRONO":    (42.466, -2.445),
        "EM06":       (42.310, -2.720),
        "EM11":       (42.250, -2.560),
        "MEDIO_OUT":  (42.225, -1.760),
        "ARGA_OUT":   (42.270, -1.800),
        "EM25":       (42.820, -1.650),
        "EM30":       (42.620, -1.650),
        "EM75_HEAD":  (42.990, -1.250),
        "IRATI_OUT":  (42.575, -1.285),
        "EM29_HEAD":  (42.770, -0.520),
        "ARAGON_OUT": (42.240, -1.760),
        "EM71":       (42.210, -1.900),
        "TUDELA":     (42.065, -1.607),
    }
    res_latlon = {
        "E001": (43.016, -4.045),
        "E029": (42.615, -1.170),
        "E075": (42.785, -1.360),
    }

    # Especificaciones físicas de los embalses (SAIH / fuentes públicas;
    # ver §1.5 del paper). Por ahora no se enchufan a la dinámica
    # (S_k^max no se impone como constraint), pero quedan disponibles
    # para extensiones futuras.
    reservoir_specs = {
        "E001": {"name": "Ebro",
                  "river": "Ebro",
                  "capacity_hm3": 540.597,
                  "cota_coronacion_m": 839.5},
        "E029": {"name": "Yesa",
                  "river": "Aragón",
                  "capacity_hm3": 446.86,
                  "cota_minima_m": 435.00,
                  "cota_nmn_m": 488.61,
                  "cota_coronacion_m": 490.0},
        "E075": {"name": "Itoiz",
                  "river": "Irati",
                  "capacity_hm3": 417.47,
                  "cota_minima_m": 485.00,
                  "cota_nmn_m": 588.00,
                  "cota_coronacion_m": 592.0},
    }

    return BasinGraph(
        type1_names=type1_names,
        edge_index_11=np.stack([s11, d11], axis=0),
        res_names=res_names,
        src12=src12,
        dst12=dst12,
        src21=src21,
        dst21=dst21,
        target_node_idx=name2idx["TUDELA"],
        rain_to_type1=rain_to_type1,
        res_to_observed={"E001": "E001", "E029": "E029", "E075": "E075"},
        edge_len_km_11=edge_len_km_11,
        len_12=len_12,
        len_21=len_21,
        type1_latlon=type1_latlon,
        res_latlon=res_latlon,
        reservoir_specs=reservoir_specs,
    )
