from typing import List, Tuple, Type
from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
class Persona(BaseModel):
    name: str
    role_description: str

class ThemesPersonasInput(BaseModel):
    themes: List[str]
    personas: List[Persona]

class PersonaThemesMapping(BaseModel):
    mapping: dict

class ThemesPersonasMatchingPrompt(PydanticPrompt[ThemesPersonasInput, PersonaThemesMapping]):
    instruction: str = "Учитывая список тем и персонажей с их ролями, сопоставьте каждого персонажа с релевантными темами на основе описания их роли."
    input_model: Type[ThemesPersonasInput] = ThemesPersonasInput
    output_model: Type[PersonaThemesMapping] = PersonaThemesMapping
    examples: List[Tuple[ThemesPersonasInput, PersonaThemesMapping]] = [
        (
            ThemesPersonasInput(
                themes=["Техническое обслуживание", "Эффективность", "Управление ресурсами", "Безопасность", "Использование GPS", "Мониторинг урожайности"],
                personas=[
                    Persona(
                        name="Технический специалист",
                        role_description="Отвечает за регулярное техническое обслуживание и проверку оборудования."
                    ),
                    Persona(
                        name="Оператор комбайна",
                        role_description="Управляет комбайном в поле, используя системы GPS и мониторинг урожайности."
                    ),
                ],
            ),
            PersonaThemesMapping(
                mapping={
                    "Технический специалист": ["Техническое обслуживание", "Безопасность", "Управление ресурсами"],
                    "Оператор комбайна": ["Эффективность", "Использование GPS", "Мониторинг урожайности"]
                }
            ),
        ),
        (
            ThemesPersonasInput(
                themes=["Калибровка оборудования", "Управление данными", "Оптимизация маршрутов", "Поддержка пользователей", "Интеграция систем"],
                personas=[
                    Persona(
                        name="Инженер по калибровке",
                        role_description="Занимается точной калибровкой весов и сенсоров для обеспечения точности измерений."
                    ),
                    Persona(
                        name="Менеджер по работе с клиентами",
                        role_description="Обеспечивает поддержку пользователей и помогает с интеграцией систем AMAZONE в рабочие процессы."
                    ),
                ],
            ),
            PersonaThemesMapping(
                mapping={
                    "Инженер по калибровке": ["Калибровка оборудования", "Оптимизация маршрутов"],
                    "Менеджер по работе с клиентами": ["Управление данными", "Поддержка пользователей", "Интеграция систем"]
                }
            ),
        ),
    ]