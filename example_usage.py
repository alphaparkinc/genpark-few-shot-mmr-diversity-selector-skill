"""Example usage for Few-Shot MMR Diversity Selector."""
from client import FewShotMmrDiversitySelector
import json

exemplar_pool = [
    {"input": "How to handle JWT expired errors in Flask?", "output": "Catch ExpiredSignatureError and return 401"},
    {"input": "Flask refresh tokens implementation", "output": "Store refresh tokens in Redis with TTL"},
    {"input": "Deploy Docker container to AWS ECS", "output": "Define task definition JSON and run service update"},
    {"input": "PostgreSQL indexing on jsonb fields", "output": "Create GIN index on jsonb column using jsonb_path_ops"}
]

query = "Best practices for JWT token authentication and session expiry"
selected = FewShotMmrDiversitySelector.select_exemplars(query, exemplar_pool, k=2, lambda_param=0.6)
print(json.dumps(selected, indent=2))
