from fabrique.loading import LoadConfig
from fabrique.models.bert.modeling import ModelArgs, Transformer
from fabrique.models.mpnet.load_rules import RULES


LOAD_CONFIG = LoadConfig(
    model_types=["mpnet"],
    model_args_class=ModelArgs,
    model_class=Transformer,
    rules=RULES,
)
