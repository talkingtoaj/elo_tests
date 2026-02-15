"""Configuration management for ELO assessment framework."""

import json
from typing import Dict, Any


class AssessmentConfig:
    """Centralized configuration for ELO assessment framework simulations."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize configuration with defaults or override from dict."""
        self._defaults = {
            # ELO system parameters
            'elo_min': 0,
            'elo_max': 10000,
            'elo_level_width': 1000,
            
            # Convergence criteria
            'variance_threshold': 500,
            'elo_error_threshold': 500,
            'max_rounds': 100,
            
            # Probability model parameters (default theoretical values)
            'prob_steepness_upper': 1.0,
            'prob_steepness_lower': 1.0,
            'prob_offset': 0.0,
            
            # Population parameters
            'entries_per_1000': 12,
            
            # Simulation parameters
            'random_state': None,
            'allow_duplicate_points': True,
        }
        
        if config_dict:
            self._config = self._defaults.copy()
            self._config.update(config_dict)
        else:
            self._config = self._defaults.copy()
    
    @property
    def elo_min(self) -> int:
        return self._config['elo_min']
    
    @property
    def elo_max(self) -> int:
        return self._config['elo_max']
    
    @property
    def elo_level_width(self) -> int:
        return self._config['elo_level_width']
    
    @property
    def variance_threshold(self) -> int:
        return self._config['variance_threshold']
    
    @property
    def elo_error_threshold(self) -> int:
        return self._config['elo_error_threshold']
    
    @property
    def max_rounds(self) -> int:
        return self._config['max_rounds']
    
    @property
    def prob_steepness_upper(self) -> float:
        return self._config['prob_steepness_upper']
    
    @property
    def prob_steepness_lower(self) -> float:
        return self._config['prob_steepness_lower']
    
    @property
    def prob_offset(self) -> float:
        return self._config['prob_offset']
    
    @property
    def entries_per_1000(self) -> int:
        return self._config['entries_per_1000']
    
    @property
    def random_state(self) -> int:
        return self._config['random_state']
    
    @property
    def allow_duplicate_points(self) -> bool:
        return self._config['allow_duplicate_points']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        if key in self._config:
            self._config[key] = value
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    
    def update(self, config_dict: Dict[str, Any]):
        """Update multiple configuration values."""
        for key, value in config_dict.items():
            if key in self._config:
                self._config[key] = value
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def get_probability_model(self):
        """Create and return a ProbabilityModel instance with current parameters."""
        from utils import ProbabilityModel
        
        return ProbabilityModel(
            steepness_upper=self.prob_steepness_upper,
            steepness_lower=self.prob_steepness_lower,
            offset=self.prob_offset
        )
    
    def save_to_file(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        config_to_save = self._config.copy()
        
        # Convert None to null for JSON
        config_to_save = {k: v for k, v in config_to_save.items() if v is not None}
        
        with open(filepath, 'w') as f:
            json.dump(config_to_save, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Returns:
            AssessmentConfig instance with loaded values
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def get_summary(self) -> str:
        """Return formatted summary of current configuration."""
        summary = []
        summary.append("ELO Assessment Framework Configuration")
        summary.append("=" * 50)
        
        # ELO system
        summary.append("\nELO System Parameters:")
        summary.append(f"  Range: {self.elo_min} - {self.elo_max}")
        summary.append(f"  Level width: {self.elo_level_width}")
        
        # Convergence
        summary.append("\nConvergence Criteria:")
        summary.append(f"  Variance threshold: {self.variance_threshold}")
        summary.append(f"  ELO error threshold: {self.elo_error_threshold}")
        summary.append(f"  Max rounds: {self.max_rounds}")
        
        # Probability model
        summary.append("\nProbability Model:")
        summary.append(f"  Steepness upper: {self.prob_steepness_upper}")
        summary.append(f"  Steepness lower: {self.prob_steepness_lower}")
        summary.append(f"  Offset: {self.prob_offset}")
        
        # Population
        summary.append(f"\nPopulation: {self.entries_per_1000} entries per 1000 ELO")
        
        return "\n".join(summary)
    
    def __repr__(self) -> str:
        return f"AssessmentConfig({self._config})"


# Convenience instance for default configuration
default_config = AssessmentConfig()
