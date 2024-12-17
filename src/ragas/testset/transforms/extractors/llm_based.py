import typing as t
from typing import Type, List, Tuple
from dataclasses import dataclass

from pydantic import BaseModel

from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.graph import Node
from ragas.testset.transforms.base import LLMBasedExtractor


class TextWithExtractionLimit(BaseModel):
    text: str
    max_num: int = 10


class SummaryExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = "Сократите данный текст до менее чем 10 предложений."
    input_model: Type[StringIO] = StringIO
    output_model: Type[StringIO] = StringIO
    examples: List[Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text="""\
                Инструкция по эксплуатации комбайна AMAZONE AFS 800

                Комбайн AMAZONE AFS 800 предназначен для эффективной уборки зерновых культур. Перед началом эксплуатации убедитесь, что уровень масла в двигателе соответствует требованиям (80 литров). Комбайн оснащен системой GPS для точного позиционирования на поле, что позволяет оптимизировать маршруты уборки и повышать производительность. Вес техники составляет 12 тонн, обеспечивая стабильность на различных типах грунта. Для управления и мониторинга параметров работы используется программное обеспечение AMAZONE Control, которое позволяет загружать карты полей и анализировать данные в реальном времени. Сенсоры YieldMaster интегрированы для измерения урожайности и помогают в принятии решений по управлению ресурсами. Регулярное техническое обслуживание, включая проверку масла и работу системы GPS, гарантирует долгий срок службы оборудования и высокую эффективность работы.
                """
            ),
            StringIO(
                text="""\
                Комбайн AMAZONE AFS 800 обеспечивает эффективную уборку зерновых культур с помощью системы GPS и сенсоров YieldMaster. Перед эксплуатацией необходимо проверить уровень масла (80 литров). Вес техники — 12 тонн, что гарантирует стабильность на различных грунтах. Управление осуществляется через ПО AMAZONE Control, позволяющее загружать карты полей и анализировать данные в реальном времени. Регулярное техническое обслуживание поддерживает высокую эффективность и долговечность оборудования.
                """
            ),
        ),
        (
            StringIO(
                text="""\
                Руководство по обслуживанию трактора AMAZONE MCX 500

                Трактор AMAZONE MCX 500 разработан для выполнения широкого спектра сельскохозяйственных работ. Регулярная проверка давления в шинах должна проводиться каждую неделю для обеспечения оптимальной производительности и безопасности. Для точной калибровки весов используется модель AMAZONE ScalePro 200, которая позволяет получать точные измерения веса прицепов и навесного оборудования. Карты обработанных территорий сохраняются в системе AMAZONE FieldManager, что облегчает планирование и анализ выполненных работ. Метрики эффективности включают расход топлива и время работы двигателя, что помогает в оптимизации использования ресурсов. Вес трактора составляет 10 тонн, что позволяет работать на крутых склонах без потери мощности. Программное обеспечение AMAZONE FieldManager интегрируется с сенсорами трактора для мониторинга состояния техники и своевременного выявления возможных неполадок.
                """
            ),
            StringIO(
                text="""\
                Трактор AMAZONE MCX 500 предназначен для разнообразных сельскохозяйственных задач с весом 10 тонн, что позволяет работать на крутых склонах. Регулярная проверка давления в шинах осуществляется каждую неделю для поддержания производительности и безопасности. Для калибровки весов используется AMAZONE ScalePro 200, обеспечивающий точные измерения. Карты обработанных территорий сохраняются в системе AMAZONE FieldManager, а метрики эффективности, такие как расход топлива и время работы двигателя, помогают оптимизировать использование ресурсов. Программное обеспечение интегрируется с сенсорами трактора для мониторинга состояния техники и предотвращения возможных неполадок.
                """
            ),
        ),
    ]


class Keyphrases(BaseModel):
    keyphrases: t.List[str]


class KeyphrasesExtractorPrompt(PydanticPrompt[TextWithExtractionLimit, Keyphrases]):
    instruction: str = "Извлеките топ-ключевых фраз, не превышающих количество max_num, из данного текста."
    input_model: Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: Type[Keyphrases] = Keyphrases
    examples: List[Tuple[TextWithExtractionLimit, Keyphrases]] = [
        (
            TextWithExtractionLimit(
                text="""\
                Инструкция по эксплуатации комбайна AMAZONE AFS 800:
                Для начала работы убедитесь, что уровень масла в двигателе составляет 80 литров. 
                Комбайн оснащен системой GPS для точного позиционирования на поле. 
                Вес техники составляет 12 тонн, что обеспечивает стабильность на различных типах грунта. 
                Карты полей можно загрузить через программное обеспечение AMAZONE Control. 
                Для измерения урожайности используются сенсоры типа YieldMaster.
                """,
                max_num=5,
            ),
            Keyphrases(
                keyphrases=[
                    "инструкция по эксплуатации",
                    "уровень масла",
                    "система GPS",
                    "программное обеспечение AMAZONE Control",
                    "сенсоры YieldMaster",
                ]
            ),
        ),
        (
            TextWithExtractionLimit(
                text="""\
                Обслуживание трактора AMAZONE MCX 500:
                Регулярная проверка давления в шинах должна проводиться каждую неделю. 
                Для точной калибровки весов используйте модель AMAZONE ScalePro 200. 
                Карты обработанных территорий сохраняются в системе AMAZONE FieldManager. 
                Метрики эффективности включают расход топлива и время работы двигателя. 
                Вес трактора составляет 10 тонн, что позволяет работать на крутых склонах.
                """,
                max_num=5,
            ),
            Keyphrases(
                keyphrases=[
                    "обслуживание трактора",
                    "проверка давления в шинах",
                    "AMAZONE ScalePro 200",
                    "AMAZONE FieldManager",
                    "метрики эффективности",
                ]
            ),
        ),
    ]

class TitleExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = "Извлеките заголовок из данного документа."
    input_model: Type[StringIO] = StringIO
    output_model: Type[StringIO] = StringIO
    examples: List[Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text="""\
                Инструкция по эксплуатации комбайна AMAZONE AFS 800

                Введение

                Эта инструкция предназначена для эксплуатации комбайна AMAZONE AFS 800. Она содержит информацию о настройке, эксплуатации и техническом обслуживании оборудования.
                """
            ),
            StringIO(text="Инструкция по эксплуатации комбайна AMAZONE AFS 800"),
        ),
        (
            StringIO(
                text="""\
                Руководство по обслуживанию трактора AMAZONE MCX 500

                Безопасность

                Перед началом обслуживания убедитесь, что трактор выключен и находится в безопасном положении.
                """
            ),
            StringIO(text="Руководство по обслуживанию трактора AMAZONE MCX 500"),
        ),
    ]


class Headlines(BaseModel):
    headlines: t.List[str]


class HeadlinesExtractorPrompt(PydanticPrompt[TextWithExtractionLimit, Headlines]):
    instruction: str = (
        "Извлеките наиболее важные заголовки, не превышающие количество max_num, из данного текста, которые могут быть использованы для разделения текста на независимые разделы. "
        "Сосредоточьтесь на заголовках уровня 2 и уровня 3."
    )

    input_model: Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: Type[Headlines] = Headlines
    examples: List[Tuple[TextWithExtractionLimit, Headlines]] = [
        (
            TextWithExtractionLimit(
                text="""\
                Введение
                Обзор техники AMAZONE и ее применение в сельском хозяйстве.

                Установка оборудования
                Пошаговая инструкция по установке техники на рабочем месте.

                Эксплуатация
                Основные правила использования техники AMAZONE.

                Поддержка и обслуживание
                Регулярное техническое обслуживание и устранение неполадок.

                Карты и схемы
                Использование карт и схем для оптимизации работы техники.

                Метрики измерения
                Ключевые показатели эффективности и их мониторинг.

                Заключение
                Резюме и рекомендации по использованию техники AMAZONE.
                """,
                max_num=7,
            ),
            Headlines(
                headlines=[
                    "Введение",
                    "Установка оборудования",
                    "Эксплуатация",
                    "Поддержка и обслуживание",
                    "Карты и схемы",
                    "Метрики измерения",
                    "Заключение",
                ],
            ),
        ),
        (
            TextWithExtractionLimit(
                text="""\
                Безопасность при работе с техникой
                Меры предосторожности и правила безопасности.

                Настройка параметров
                Инструкция по настройке ключевых параметров техники.

                Калибровка весов
                Пошаговая процедура калибровки весов для точных измерений.

                Использование сенсоров
                Настройка и калибровка сенсоров для оптимальной работы.

                Программное обеспечение
                Обзор программных инструментов для управления техникой.

                Обновления и апгрейды
                Процесс обновления программного обеспечения и аппаратных компонентов.

                Часто задаваемые вопросы
                Ответы на наиболее распространенные вопросы пользователей.
                """,
                max_num=7,
            ),
            Headlines(
                headlines=[
                    "Безопасность при работе с техникой",
                    "Настройка параметров",
                    "Калибровка весов",
                    "Использование сенсоров",
                    "Программное обеспечение",
                    "Обновления и апгрейды",
                    "Часто задаваемые вопросы",
                ],
            ),
        ),
    ]


class NEROutput(BaseModel):
    entities: t.List[str]


class NERPrompt(PydanticPrompt[TextWithExtractionLimit, NEROutput]):
    instruction: str = (
        "Извлеките именованные сущности из данного текста, ограничивая вывод наиболее значимыми сущностями. "
        "Убедитесь, что количество сущностей не превышает указанного максимума."
    )
    
    input_model: Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: Type[NEROutput] = NEROutput
    examples: List[Tuple[TextWithExtractionLimit, NEROutput]] = [
        (
            TextWithExtractionLimit(
                text="""\
                Инструкция по эксплуатации комбайна AMAZONE AFS 800:
                Для начала работы убедитесь, что уровень масла в двигателе составляет 80 литров. 
                Комбайн оснащен системой GPS для точного позиционирования на поле. 
                Вес техники составляет 12 тонн, что обеспечивает стабильность на различных типах грунта. 
                Карты полей можно загрузить через программное обеспечение AMAZONE Control. 
                Для измерения урожайности используются сенсоры типа YieldMaster.
                """,
                max_num=10,
            ),
            NEROutput(
                entities=[
                    "комбайн AMAZONE AFS 800",
                    "уровень масла",
                    "80 литров",
                    "GPS",
                    "12 тонн",
                    "различных типах грунта",
                    "карты полей",
                    "программное обеспечение AMAZONE Control",
                    "урожайность",
                    "сенсоры типа YieldMaster",
                ]
            ),
        ),
        (
            TextWithExtractionLimit(
                text="""\
                Обслуживание трактора AMAZONE MCX 500:
                Регулярная проверка давления в шинах должна проводиться каждую неделю. 
                Для точной калибровки весов используйте модель AMAZONE ScalePro 200. 
                Карты обработанных территорий сохраняются в системе AMAZONE FieldManager. 
                Метрики эффективности включают расход топлива и время работы двигателя. 
                Вес трактора составляет 10 тонн, что позволяет работать на крутых склонах.
                """,
                max_num=10,
            ),
            NEROutput(
                entities=[
                    "трактора AMAZONE MCX 500",
                    "давления в шинах",
                    "еждую неделю",
                    "калибровки весов",
                    "AMAZONE ScalePro 200",
                    "карты обработанных территорий",
                    "AMAZONE FieldManager",
                    "метрики эффективности",
                    "расход топлива",
                    "10 тонн",
                ]
            ),
        ),
    ]


@dataclass
class SummaryExtractor(LLMBasedExtractor):
    """
    Extracts a summary from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : SummaryExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "summary"
    prompt: SummaryExtractorPrompt = SummaryExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        result = await self.prompt.generate(self.llm, data=StringIO(text=chunks[0]))
        return self.property_name, result.text


@dataclass
class KeyphrasesExtractor(LLMBasedExtractor):
    """
    Extracts top keyphrases from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : KeyphrasesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "keyphrases"
    prompt: KeyphrasesExtractorPrompt = KeyphrasesExtractorPrompt()
    max_num: int = 5

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        keyphrases = []
        for chunk in chunks:
            result = await self.prompt.generate(
                self.llm, data=TextWithExtractionLimit(text=chunk, max_num=self.max_num)
            )
            keyphrases.extend(result.keyphrases)
        return self.property_name, keyphrases


@dataclass
class TitleExtractor(LLMBasedExtractor):
    """
    Extracts the title from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : TitleExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "title"
    prompt: TitleExtractorPrompt = TitleExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        result = await self.prompt.generate(self.llm, data=StringIO(text=chunks[0]))
        return self.property_name, result.text


@dataclass
class HeadlinesExtractor(LLMBasedExtractor):
    """
    Extracts the headlines from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : HeadlinesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "headlines"
    prompt: HeadlinesExtractorPrompt = HeadlinesExtractorPrompt()
    max_num: int = 5

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        headlines = []
        for chunk in chunks:
            result = await self.prompt.generate(
                self.llm, data=TextWithExtractionLimit(text=chunk, max_num=self.max_num)
            )
            if result:
                headlines.extend(result.headlines)
        return self.property_name, headlines


@dataclass
class NERExtractor(LLMBasedExtractor):
    """
    Extracts named entities from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract. Defaults to "entities".
    prompt : NERPrompt
        The prompt used for extraction.
    """

    property_name: str = "entities"
    prompt: PydanticPrompt[TextWithExtractionLimit, NEROutput] = NERPrompt()
    max_num_entities: int = 10

    async def extract(self, node: Node) -> t.Tuple[str, t.List[str]]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, []
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        entities = []
        for chunk in chunks:
            result = await self.prompt.generate(
                self.llm,
                data=TextWithExtractionLimit(text=chunk, max_num=self.max_num_entities),
            )
            entities.extend(result.entities)
        return self.property_name, entities


class TopicDescription(BaseModel):
    description: str


class TopicDescriptionPrompt(PydanticPrompt[StringIO, TopicDescription]):
    instruction: str = "Предоставьте краткое описание основных тем, обсуждаемых в следующем тексте."
    input_model: Type[StringIO] = StringIO
    output_model: Type[TopicDescription] = TopicDescription
    examples: List[Tuple[StringIO, TopicDescription]] = [
        (
            StringIO(
                text="""\
                Инструкция по эксплуатации комбайна AMAZONE AFS 800

                Комбайн AMAZONE AFS 800 предназначен для эффективной уборки зерновых культур. Перед началом эксплуатации убедитесь, что уровень масла в двигателе соответствует требованиям (80 литров). Комбайн оснащен системой GPS для точного позиционирования на поле, что позволяет оптимизировать маршруты уборки и повышать производительность. Вес техники составляет 12 тонн, обеспечивая стабильность на различных типах грунта. Для управления и мониторинга параметров работы используется программное обеспечение AMAZONE Control, которое позволяет загружать карты полей и анализировать данные в реальном времени. Сенсоры YieldMaster интегрированы для измерения урожайности и помогают в принятии решений по управлению ресурсами. Регулярное техническое обслуживание, включая проверку масла и работу системы GPS, гарантирует долгий срок службы оборудования и высокую эффективность работы.
                """
            ),
            TopicDescription(
                description="Описание комбайна AMAZONE AFS 800, включая его технические характеристики, систему GPS, программное обеспечение AMAZONE Control, сенсоры YieldMaster и требования к техническому обслуживанию для обеспечения эффективности и долговечности."
            ),
        ),
        (
            StringIO(
                text="""\
                Руководство по обслуживанию трактора AMAZONE MCX 500

                Трактор AMAZONE MCX 500 разработан для выполнения широкого спектра сельскохозяйственных работ. Регулярная проверка давления в шинах должна проводиться каждую неделю для обеспечения оптимальной производительности и безопасности. Для точной калибровки весов используется модель AMAZONE ScalePro 200, которая позволяет получать точные измерения веса прицепов и навесного оборудования. Карты обработанных территорий сохраняются в системе AMAZONE FieldManager, что облегчает планирование и анализ выполненных работ. Метрики эффективности включают расход топлива и время работы двигателя, что помогает в оптимизации использования ресурсов. Вес трактора составляет 10 тонн, что позволяет работать на крутых склонах без потери мощности. Программное обеспечение AMAZONE FieldManager интегрируется с сенсорами трактора для мониторинга состояния техники и своевременного выявления возможных неполадок.
                """
            ),
            TopicDescription(
                description="Обслуживание трактора AMAZONE MCX 500, включая регулярную проверку давления в шинах, калибровку весов с помощью AMAZONE ScalePro 200, использование системы AMAZONE FieldManager для хранения карт обработанных территорий и мониторинг метрик эффективности для оптимизации ресурсов."
            ),
        ),
    ]


@dataclass
class TopicDescriptionExtractor(LLMBasedExtractor):
    """
    Extracts a concise description of the main topic(s) discussed in the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : TopicDescriptionPrompt
        The prompt used for extraction.
    """

    property_name: str = "topic_description"
    prompt: PydanticPrompt = TopicDescriptionPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        result = await self.prompt.generate(self.llm, data=StringIO(text=chunks[0]))
        return self.property_name, result.description


class ThemesAndConcepts(BaseModel):
    output: t.List[str]


class ThemesAndConceptsExtractorPrompt(PydanticPrompt[TextWithExtractionLimit, ThemesAndConcepts]):
    instruction: str = "Извлеките основные темы и концепции из данного текста."
    input_model: Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: Type[ThemesAndConcepts] = ThemesAndConcepts
    examples: List[Tuple[TextWithExtractionLimit, ThemesAndConcepts]] = [
        (
            TextWithExtractionLimit(
                text="""\
                Инструкция по эксплуатации комбайна AMAZONE AFS 800

                Комбайн AMAZONE AFS 800 предназначен для эффективной уборки зерновых культур. Перед началом эксплуатации убедитесь, что уровень масла в двигателе соответствует требованиям (80 литров). Комбайн оснащен системой GPS для точного позиционирования на поле, что позволяет оптимизировать маршруты уборки и повышать производительность. Вес техники составляет 12 тонн, обеспечивая стабильность на различных типах грунта. Для управления и мониторинга параметров работы используется программное обеспечение AMAZONE Control, которое позволяет загружать карты полей и анализировать данные в реальном времени. Сенсоры YieldMaster интегрированы для измерения урожайности и помогают в принятии решений по управлению ресурсами. Регулярное техническое обслуживание, включая проверку масла и работу системы GPS, гарантирует долгий срок службы оборудования и высокую эффективность работы.
                """,
                max_num=10,
            ),
            ThemesAndConcepts(
                output=[
                    "Эксплуатация комбайна",
                    "Технические характеристики",
                    "Система GPS",
                    "Программное обеспечение AMAZONE Control",
                    "Сенсоры YieldMaster",
                    "Уровень масла",
                    "Оптимизация маршрутов уборки",
                    "Мониторинг параметров работы",
                    "Техническое обслуживание",
                    "Урожайность"
                ]
            ),
        ),
        (
            TextWithExtractionLimit(
                text="""\
                Руководство по обслуживанию трактора AMAZONE MCX 500

                Трактор AMAZONE MCX 500 разработан для выполнения широкого спектра сельскохозяйственных работ. Регулярная проверка давления в шинах должна проводиться каждую неделю для обеспечения оптимальной производительности и безопасности. Для точной калибровки весов используется модель AMAZONE ScalePro 200, которая позволяет получать точные измерения веса прицепов и навесного оборудования. Карты обработанных территорий сохраняются в системе AMAZONE FieldManager, что облегчает планирование и анализ выполненных работ. Метрики эффективности включают расход топлива и время работы двигателя, что помогает в оптимизации использования ресурсов. Вес трактора составляет 10 тонн, что позволяет работать на крутых склонах без потери мощности. Программное обеспечение AMAZONE FieldManager интегрируется с сенсорами трактора для мониторинга состояния техники и своевременного выявления возможных неполадок.
                """,
                max_num=10,
            ),
            ThemesAndConcepts(
                output=[
                    "Обслуживание трактора",
                    "Проверка давления в шинах",
                    "Калибровка весов",
                    "Модель AMAZONE ScalePro 200",
                    "Система AMAZONE FieldManager",
                    "Метрики эффективности",
                    "Расход топлива",
                    "Время работы двигателя",
                    "Вес трактора",
                    "Мониторинг состояния техники"
                ]
            ),
        ),
    ]


@dataclass
class ThemesExtractor(LLMBasedExtractor):
    """
    Extracts themes from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract. Defaults to "themes".
    prompt : ThemesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "themes"
    prompt: ThemesAndConceptsExtractorPrompt = ThemesAndConceptsExtractorPrompt()
    max_num_themes: int = 10

    async def extract(self, node: Node) -> t.Tuple[str, t.List[str]]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, []
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        themes = []
        for chunk in chunks:
            result = await self.prompt.generate(
                self.llm,
                data=TextWithExtractionLimit(text=chunk, max_num=self.max_num_themes),
            )
            themes.extend(result.output)

        return self.property_name, themes
