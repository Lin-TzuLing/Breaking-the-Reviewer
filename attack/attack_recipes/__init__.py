from .BertAttack import BertAttack
from .DeepwordBug import DeepWordBug
from .PuncAttack import PuncAttack
from .StyleAdv import StyleAdv
from .TextFooler import TextFooler

ATTACK_RECIPES = {
    "textfooler": TextFooler,
    "deepwordbug": DeepWordBug,
    "puncattack": PuncAttack,
    "bertattack": BertAttack,
    "styleadv": StyleAdv,
}


def load_attack_recipe(attack_name, attack_config):
    attack_recipe = ATTACK_RECIPES[attack_name](attack_config)
    print(f"\n Attack Recipe: {attack_recipe.__class__.__name__} loaded successfully!")
    return attack_recipe
