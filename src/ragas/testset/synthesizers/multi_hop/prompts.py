import typing as t
from typing import List, Tuple, Type
from pydantic import BaseModel, Field

from ragas.prompt import PydanticPrompt
from ragas.testset.persona import Persona


class ConceptsList(BaseModel):
    lists_of_concepts: t.List[t.List[str]] = Field(
        description="A list containing lists of concepts from each node"
    )
    max_combinations: int = Field(
        description="The maximum number of concept combinations to generate", default=5
    )


class ConceptCombinations(BaseModel):
    combinations: t.List[t.List[str]]


class ConceptCombinationPrompt(PydanticPrompt[ConceptsList, ConceptCombinations]):
    instruction: str = (
        "Сформируйте комбинации, объединяя концепции из как минимум двух разных списков.\n"
        "**Инструкции:**\n"
        "- Просмотрите концепции из каждого узла.\n"
        "- Определите концепции, которые могут быть логически связаны или противопоставлены.\n"
        "- Сформируйте комбинации, включающие концепции из разных узлов.\n"
        "- Каждая комбинация должна содержать как минимум одну концепцию из двух или более узлов.\n"
        "- Перечислите комбинации ясно и лаконично.\n"
        "- Не повторяйте одну и ту же комбинацию более одного раза."
    )
    input_model: Type[ConceptsList] = ConceptsList  # Содержит списки концепций из каждого узла
    output_model: Type[ConceptCombinations] = ConceptCombinations  # Содержит список комбинаций концепций
    examples: List[Tuple[ConceptsList, ConceptCombinations]] = [
        (
            ConceptsList(
                lists_of_concepts=[
                    ["Техническое обслуживание", "Эффективность"],  # Концепции из Узла 1
                    ["Использование GPS", "Мониторинг урожайности"],  # Концепции из Узла 2
                ],
                max_combinations=2,
            ),
            ConceptCombinations(
                combinations=[
                    ["Техническое обслуживание", "Использование GPS"],
                    ["Эффективность", "Мониторинг урожайности"],
                ]
            ),
        ),
        (
            ConceptsList(
                lists_of_concepts=[
                    ["Оптимизация маршрутов", "Безопасность"],  # Концепции из Узла 1
                    ["Интеграция систем", "Управление ресурсами"],  # Концепции из Узла 2
                    ["Калибровка оборудования", "Поддержка пользователей"],  # Концепции из Узла 3
                ],
                max_combinations=3,
            ),
            ConceptCombinations(
                combinations=[
                    ["Оптимизация маршрутов", "Интеграция систем"],
                    ["Безопасность", "Управление ресурсами"],
                    ["Калибровка оборудования", "Поддержка пользователей"],
                ]
            ),
        ),
    ]

class QueryConditions(BaseModel):
    persona: Persona
    themes: t.List[str]
    query_style: str
    query_length: str
    context: t.List[str]


class GeneratedQueryAnswer(BaseModel):
    query: str
    answer: str


class QueryAnswerGenerationPrompt(
    PydanticPrompt[QueryConditions, GeneratedQueryAnswer]
):
    instruction: str = (
        "Сгенерируйте запрос и ответ на основе заданных условий (персона, темы, стиль, длина) "
        "и предоставленного контекста. Убедитесь, что ответ полностью соответствует контексту, используя только информацию "
        "прямо из предоставленных узлов."
        "### Инструкции:\n\n"
        "1. **Сгенерировать запрос**: Исходя из контекста, персоны, тем, стиля и длины, создайте вопрос, "
        "соответствующий перспективе персоны и отражающий темы.\n"
        "2. **Сгенерировать ответ**: Используя только содержимое предоставленного контекста, создайте точный и подробный ответ на "
        "запрос. Не включайте никакой информации, которая отсутствует в контексте или не может быть выведена из него.\n"
        "### Пример выходных данных:\n\n"
    )
    input_model: Type[QueryConditions] = QueryConditions
    output_model: Type[GeneratedQueryAnswer] = GeneratedQueryAnswer
    examples: List[Tuple[QueryConditions, GeneratedQueryAnswer]] = [
        (
            QueryConditions(
                persona=Persona(
                    name="Технический специалист",
                    role_description="Отвечает за техническое обслуживание и настройку оборудования AMAZONE."
                ),
                themes=["Техническое обслуживание", "Использование GPS", "Мониторинг урожайности"],
                query_style="Формальный",
                query_length="Краткий",
                context=[
                    "Комбайн AMAZONE AFS 800 оснащен системой GPS для точного позиционирования на поле.",
                    "Регулярное техническое обслуживание включает проверку уровня масла и работу сенсоров YieldMaster для мониторинга урожайности.",
                    "Использование GPS позволяет оптимизировать маршруты уборки и повышать эффективность работы комбайна."
                ]
            ),
            GeneratedQueryAnswer(
                query="Какие ключевые аспекты включает техническое обслуживание комбайна AMAZONE AFS 800?",
                answer="Техническое обслуживание комбайна AMAZONE AFS 800 включает проверку уровня масла, работу сенсоров YieldMaster для мониторинга урожайности и обеспечение функционирования системы GPS для оптимизации маршрутов уборки."
            ),
        ),
        (
            QueryConditions(
                persona=Persona(
                    name="Менеджер по работе с клиентами",
                    role_description="Обеспечивает поддержку пользователей и отвечает на их вопросы по эксплуатации техники AMAZONE."
                ),
                themes=["Поддержка пользователей", "Интеграция систем", "Обучение операторов"],
                query_style="Дружелюбный",
                query_length="Средний",
                context=[
                    "Менеджеры по работе с клиентами обеспечивают поддержку пользователей техники AMAZONE, помогая с интеграцией систем AMAZONE Control и проведением обучающих сессий для операторов.",
                    "Интеграция систем позволяет пользователям эффективно управлять параметрами оборудования и получать своевременные обновления программного обеспечения.",
                    "Обучение операторов включает практические занятия по использованию сенсоров YieldMaster и систем мониторинга урожайности."
                ]
            ),
            GeneratedQueryAnswer(
                query="Как менеджеры по работе с клиентами помогают пользователям эффективно использовать технику AMAZONE?",
                answer="Менеджеры по работе с клиентами помогают пользователям эффективно использовать технику AMAZONE, обеспечивая поддержку при интеграции систем AMAZONE Control, проводя обучающие сессии для операторов и предоставляя помощь в использовании сенсоров YieldMaster и систем мониторинга урожайности."
            ),
        ),
    ]
