"""
core/db_artifacts.py
Gestione degli artefatti del modello su MongoDB (GridFS).
Stub per sviluppo locale senza MongoDB.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

log = logging.getLogger(__name__)


def save_model_artifacts(db, artifacts_dict: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
    """
    Salva artefatti del modello su MongoDB GridFS.
    Ritorna la versione generata (es. 'v20250303_1200') o None in caso di fallimento.
    """
    if db is None:
        log.warning("MongoDB non disponibile. Salvataggio artefatti skippato.")
        return None
    
    try:
        import pickle
        from gridfs import GridFS
        version = datetime.now().strftime("v%Y%m%d_%H%M")
        fs = GridFS(db, collection="model_artifacts")
        
        # Salva ogni artefatto come file separato
        for key, data in artifacts_dict.items():
            filename = f"{version}_{key}.pkl"
            existing = fs.find_one({"filename": filename})
            if existing:
                fs.delete(existing._id)
            # Serializza i dati in bytes
            serialized_data = pickle.dumps(data)
            fs.put(serialized_data, filename=filename, metadata={"version": version, **metadata})
        
        # Salva metadati in una collezione separata
        db.model_versions.insert_one({
            "version": version,
            "created_at": datetime.utcnow(),
            **metadata
        })
        
        log.info(f"Artefatti salvati come versione {version}")
        return version
    except Exception as e:
        log.error(f"Errore salvataggio MongoDB GridFS: {e}")
        return None


def list_model_versions(db) -> List[Dict[str, Any]]:
    """
    Lista le versioni dei modello salvate su MongoDB.
    """
    if db is None:
        log.warning("MongoDB non disponibile. Nessuna versione trovata.")
        return []
    
    try:
        versions = list(db.model_versions.find(
            {}, 
            {"_id": 0, "version": 1, "train_through_round": 1, "train_through_year": 1,
             "walk_forward_brier": 1, "kendall_tau": 1, "calibrator_fitted": 1}
        ).sort("created_at", -1))
        return versions
    except Exception as e:
        log.error(f"Errore lettura versioni: {e}")
        return []


def rollback_to_version(db, version: str) -> bool:
    """
    Ripristina una versione specifica come attiva.
    """
    if db is None:
        log.warning("MongoDB non disponibile. Rollback impossibile.")
        return False
    
    try:
        # Verifica che la versione esista
        if not db.model_versions.find_one({"version": version}):
            log.error(f"Versione {version} non trovata.")
            return False
        
        # Elimina la versione corrente (se esiste) e marca questa come attiva
        db.model_versions.update_many(
            {"active": True},
            {"$set": {"active": False}}
        )
        db.model_versions.update_one(
            {"version": version},
            {"$set": {"active": True}}
        )
        log.info(f"Rollback a versione {version} completato.")
        return True
    except Exception as e:
        log.error(f"Errore rollback: {e}")
        return False


def delete_old_versions(db, keep_last_n: int) -> int:
    """
    Elimina versioni vecchie, mantiene solo le ultime N.
    """
    if db is None:
        log.warning("MongoDB non disponibile. Eliminazione impossibile.")
        return 0
    
    try:
        # Trova le versioni più vecchie da eliminare
        all_versions = list(db.model_versions.find(
            {}, {"_id": 1, "version": 1, "created_at": 1}
        ).sort("created_at", -1))
        
        if len(all_versions) <= keep_last_n:
            return 0
        
        to_delete = all_versions[keep_last_n:]
        deleted = 0
        
        for ver in to_delete:
            # Elimina file GridFS associati
            from gridfs import GridFS
            fs = GridFS(db, collection="model_artifacts")
            for grid_file in fs.find({"metadata.version": ver["version"]}):
                fs.delete(grid_file._id)
            
            # Elimina record versione
            db.model_versions.delete_one({"_id": ver["_id"]})
            deleted += 1
        
        log.info(f"Eliminate {deleted} versioni vecchie.")
        return deleted
    except Exception as e:
        log.error(f"Errore eliminazione versioni: {e}")
        return 0