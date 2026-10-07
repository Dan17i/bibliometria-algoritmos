"""Pruebas de la exportación de fusiones a JSON (src/clustering/export.py)."""
import json

import pytest

from src.clustering.export import build_export, export_merges_json, load_export
from src.clustering.linkage import LINKAGES, hierarchical_clustering

# 4 puntos en una recta (0, 1, 5, 7), resueltos a mano en test_clustering.py
DIST = [
    [0, 1, 5, 7],
    [1, 0, 4, 6],
    [5, 4, 0, 2],
    [7, 6, 2, 0],
]
LABELS = ["p0", "p1", "p2", "p3"]


def test_json_tiene_el_formato_documentado(tmp_path):
    ruta = tmp_path / "fusiones.json"
    export_merges_json(str(ruta), DIST, LABELS)
    datos = json.loads(ruta.read_text(encoding="utf-8"))

    assert datos["version"] == 1
    assert datos["n"] == 4
    assert datos["labels"] == LABELS
    assert set(datos["linkages"]) == {"single", "complete", "average"}
    for info in datos["linkages"].values():
        assert set(info) == {"cpcc", "merges"}
        assert len(info["merges"]) == 3
        assert all(len(m) == 4 for m in info["merges"])


def test_fusiones_a_mano_en_el_archivo(tmp_path):
    ruta = tmp_path / "fusiones.json"
    export_merges_json(str(ruta), DIST, LABELS)
    ling = json.loads(ruta.read_text(encoding="utf-8"))["linkages"]

    assert ling["single"]["merges"] == [[0, 1, 1, 2], [2, 3, 2, 2], [4, 5, 4, 4]]
    assert ling["complete"]["merges"] == [[0, 1, 1, 2], [2, 3, 2, 2], [4, 5, 7, 4]]
    assert ling["average"]["merges"] == [[0, 1, 1, 2], [2, 3, 2, 2], [4, 5, 5.5, 4]]
    assert ling["single"]["cpcc"] == pytest.approx(0.8985, abs=1e-3)


def test_coincide_con_hierarchical_clustering(tmp_path):
    ruta = tmp_path / "fusiones.json"
    export_merges_json(str(ruta), DIST, LABELS)
    ling = json.loads(ruta.read_text(encoding="utf-8"))["linkages"]
    for linkage in LINKAGES:
        esperado = [list(m) for m in hierarchical_clustering(DIST, linkage)]
        assert ling[linkage]["merges"] == esperado


def test_ida_y_vuelta_y_validacion(tmp_path):
    ruta = tmp_path / "fusiones.json"
    escrito = export_merges_json(str(ruta), DIST, LABELS)
    assert load_export(str(ruta)) == escrito


def test_conserva_tildes_y_enlaces_elegidos(tmp_path):
    ruta = tmp_path / "fusiones.json"
    etiquetas = ["Visión", "Lenguaje", "Audio", "Año"]
    export_merges_json(str(ruta), DIST, etiquetas, linkages=("average",))
    texto = ruta.read_text(encoding="utf-8")
    assert "Visión" in texto  # ensure_ascii=False
    assert list(json.loads(texto)["linkages"]) == ["average"]


def test_etiquetas_con_longitud_incorrecta():
    with pytest.raises(ValueError):
        build_export(DIST, ["a", "b"])


def test_enlace_desconocido():
    with pytest.raises(ValueError):
        build_export(DIST, LABELS, linkages=("ward",))


def test_load_export_detecta_archivos_invalidos(tmp_path):
    base = build_export(DIST, LABELS)

    def guardar(datos, nombre):
        p = tmp_path / nombre
        p.write_text(json.dumps(datos), encoding="utf-8")
        return str(p)

    malo = json.loads(json.dumps(base))
    malo["version"] = 99
    with pytest.raises(ValueError):
        load_export(guardar(malo, "version.json"))

    malo = json.loads(json.dumps(base))
    malo["linkages"]["single"]["merges"].pop()
    with pytest.raises(ValueError):
        load_export(guardar(malo, "faltan.json"))

    malo = json.loads(json.dumps(base))
    malo["linkages"]["single"]["merges"][2][0] = 0  # id ya fusionado
    with pytest.raises(ValueError):
        load_export(guardar(malo, "ids.json"))

    malo = json.loads(json.dumps(base))
    malo["linkages"]["complete"]["merges"][2][2] = 0.5  # altura decreciente
    with pytest.raises(ValueError):
        load_export(guardar(malo, "alturas.json"))


def test_rechaza_nan(tmp_path):
    dist_nan = [[0, float("nan")], [float("nan"), 0]]
    with pytest.raises(ValueError):
        export_merges_json(str(tmp_path / "x.json"), dist_nan, ["a", "b"])
