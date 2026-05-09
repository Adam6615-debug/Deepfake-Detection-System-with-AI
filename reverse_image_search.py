"""
reverse_image_search.py
------------------------
Reverse image search using perceptual hashing.
Detects duplicates and near-duplicates of known fake images.
"""

import os
import json
import numpy as np
from PIL import Image
import imagehash
from pathlib import Path


class ReverseImageSearch:
    """Reverse image search using perceptual hashing."""
    
    def __init__(self, database_path: str = "data/reverse_image_db.json"):
        """
        Initialize reverse image search.
        
        Args:
            database_path: Path to hash database file
        """
        self.database_path = database_path
        self.database = self._load_database()
        
    def _load_database(self) -> dict:
        """Load hash database from file."""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not load database: {e}")
        
        # Create empty database structure
        return {
            "known_fakes": [],
            "known_reals": [],
            "metadata": {
                "created": None,
                "last_updated": None,
                "total_images": 0
            }
        }
    
    def _save_database(self):
        """Save hash database to file."""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        self.database["metadata"]["last_updated"] = str(np.datetime64('now'))
        self.database["metadata"]["total_images"] = len(self.database["known_fakes"]) + len(self.database["known_reals"])
        
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.database, f, indent=2)
            print(f"[DB] Database saved to: {self.database_path}")
        except Exception as e:
            print(f"[ERROR] Could not save database: {e}")
    
    def compute_hashes(self, image_path: str) -> dict:
        """
        Compute multiple perceptual hashes for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Dictionary of different hash types
        """
        try:
            img = Image.open(image_path)
            
            hashes = {
                "average_hash": str(imagehash.average_hash(img)),
                "perceptual_hash": str(imagehash.phash(img)),
                "difference_hash": str(imagehash.dhash(img)),
                "wavelet_hash": str(imagehash.whash(img)),
                "color_hash": str(imagehash.colorhash(img))
            }
            
            return hashes
            
        except Exception as e:
            print(f"[ERROR] Could not compute hashes for {image_path}: {e}")
            return {}
    
    def add_to_database(self, image_path: str, label: str, metadata: dict = None):
        """
        Add an image to the hash database.
        
        Args:
            image_path: Path to image file
            label: "fake" or "real"
            metadata: Optional metadata about the image
        """
        hashes = self.compute_hashes(image_path)
        if not hashes:
            return False
            
        entry = {
            "image_path": image_path,
            "label": label,
            "hashes": hashes,
            "metadata": metadata or {}
        }
        
        if label == "fake":
            self.database["known_fakes"].append(entry)
        elif label == "real":
            self.database["known_reals"].append(entry)
        else:
            print(f"[WARNING] Unknown label: {label}")
            return False
            
        self._save_database()
        print(f"[DB] Added {label} image to database: {image_path}")
        return True
    
    def build_database_from_folders(self, real_folder: str = "data/real", fake_folder: str = "data/fake"):
        """
        Build database from image folders.
        
        Args:
            real_folder: Path to real images folder
            fake_folder: Path to fake images folder
        """
        print("[DB] Building hash database from folders...")
        
        # Clear existing database
        self.database = {
            "known_fakes": [],
            "known_reals": [],
            "metadata": {
                "created": str(np.datetime64('now')),
                "last_updated": None,
                "total_images": 0
            }
        }
        
        # Add real images
        if os.path.exists(real_folder):
            for img_file in Path(real_folder).glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.add_to_database(str(img_file), "real")
        
        # Add fake images
        if os.path.exists(fake_folder):
            for img_file in Path(fake_folder).glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.add_to_database(str(img_file), "fake")
        
        print(f"[DB] Database built with {self.database['metadata']['total_images']} images")
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hash strings."""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def search_similar_images(self, image_path: str, max_distance: int = 5) -> dict:
        """
        Search for similar images in database.
        
        Args:
            image_path: Path to query image
            max_distance: Maximum Hamming distance for a genuine match
            
        Returns:
            dict: Search results with matches and similarity scores
        """
        query_hashes = self.compute_hashes(image_path)
        if not query_hashes:
            return {"matches": [], "similarity": 0.0, "best_match": None}

        # FIX 1: correct per-hash-type bit lengths.
        # average/perceptual/difference/wavelet hashes are 64-bit hex strings (16 chars).
        # color_hash is a different format entirely — exclude it from scoring because
        # all forward-facing face photos share similar skin-tone distributions, making
        # color_hash useless for identity matching and inflating similarity to ~0.9.
        HASH_BITS = {
            "average_hash":    64,
            "perceptual_hash": 64,
            "difference_hash": 64,
            "wavelet_hash":    64,
            # color_hash intentionally excluded
        }

        # FIX 2: genuine match threshold — Hamming distance ≤ max_distance on a
        # 64-bit hash means similarity ≥ (64 - max_distance) / 64 ≈ 0.92 for max_distance=5.
        # Old code used 8/64 = 0.125, which matched nearly everything.
        MATCH_THRESHOLD = (64 - max_distance) / 64.0   # ≈ 0.922 for max_distance=5

        matches = []

        # FIX 3: search BOTH known_fakes AND known_reals.
        # Old code only searched known_fakes, so a real image could never find its
        # own entry and always showed a fake as the best match.
        all_entries = (
            [(e, "fake") for e in self.database["known_fakes"]] +
            [(e, "real") for e in self.database["known_reals"]]
        )

        for entry, _ in all_entries:
            similarities = {}
            for hash_type, bits in HASH_BITS.items():
                if hash_type in query_hashes and hash_type in entry["hashes"]:
                    distance = self.hamming_distance(
                        query_hashes[hash_type], entry["hashes"][hash_type]
                    )
                    similarity = 1.0 - (distance / bits)
                    similarities[hash_type] = similarity

            if not similarities:
                continue

            avg_similarity = float(np.mean(list(similarities.values())))

            # Only keep genuine matches — skip everything below the threshold
            if avg_similarity >= MATCH_THRESHOLD:
                matches.append({
                    "image_path": entry["image_path"],
                    "label": entry["label"],
                    "similarities": similarities,
                    "avg_similarity": avg_similarity
                })

        # Sort best matches first
        matches.sort(key=lambda x: x["avg_similarity"], reverse=True)

        overall_similarity = 0.0
        best_match = None

        if matches:
            best_match = matches[0]
            overall_similarity = best_match["avg_similarity"]

        return {
            "matches": matches,
            "similarity": overall_similarity,
            "best_match": best_match,
            "total_searched": len(self.database["known_fakes"]) + len(self.database["known_reals"])
        }
    
    def analyze_image(self, image_path: str) -> dict:
        """
        Perform reverse image search analysis.
        
        Args:
            image_path: Path to image to analyze
            
        Returns:
            dict: Analysis results for report
        """
        print("[REVERSE] Searching for similar images...")
        
        results = self.search_similar_images(image_path, max_distance=8)
        
        if results["similarity"] > 0.85:
            verdict = "HIGH_SIMILARITY"
            confidence = results["similarity"] * 100
        elif results["similarity"] > 0.70:
            verdict = "MODERATE_SIMILARITY"
            confidence = results["similarity"] * 80
        elif results["similarity"] > 0.50:
            verdict = "LOW_SIMILARITY"
            confidence = results["similarity"] * 60
        else:
            verdict = "NO_MATCH"
            confidence = 0.0
        
        analysis_result = {
            "similarity": results["similarity"],
            "verdict": verdict,
            "confidence": confidence,
            "matches_found": len(results["matches"]),
            "best_match": results["best_match"],
            "database_size": results["total_searched"]
        }
        
        if results["best_match"]:
            print(f"[REVERSE] Found similar image with {results['similarity']:.3f} similarity")
        else:
            print("[REVERSE] No similar images found")
        
        return analysis_result


def analyze_with_reverse_search(image_path: str, report: dict) -> dict:
    """
    Run reverse image search and update report.
    
    Args:
        image_path: Path to input image
        report: Report dictionary to update
        
    Returns:
        dict: Updated report dictionary
    """
    # Initialize reverse image search
    searcher = ReverseImageSearch()
    
    # If database is empty, try to build it from data folders
    if searcher.database["metadata"]["total_images"] == 0:
        print("[REVERSE] Database empty, building from data folders...")
        searcher.build_database_from_folders()
    
    # Perform analysis
    results = searcher.analyze_image(image_path)
    
    # Update report
    report["reverse_image_match"] = results
    
    return report
