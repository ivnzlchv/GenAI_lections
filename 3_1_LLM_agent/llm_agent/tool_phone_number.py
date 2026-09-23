"""Инструмент для нормализации и валидации телефонных номеров."""

from typing import Dict, Optional, Union

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat, PhoneNumberType


class PhoneNumberTool:
    """Проверяет телефонные номера и приводит их к стандартному формату."""

    name = "phone_number"
    description = (
        "Нормализует и проверяет телефонный номер. "
        "Локальные номера по умолчанию интерпретируются как российские."
    )

    _FORMATS = {
        "E164": PhoneNumberFormat.E164,
        "INTERNATIONAL": PhoneNumberFormat.INTERNATIONAL,
        "NATIONAL": PhoneNumberFormat.NATIONAL,
        "RFC3966": PhoneNumberFormat.RFC3966,
    }

    _NUMBER_TYPES = {
        PhoneNumberType.FIXED_LINE: "fixed_line",
        PhoneNumberType.MOBILE: "mobile",
        PhoneNumberType.FIXED_LINE_OR_MOBILE: "fixed_line_or_mobile",
        PhoneNumberType.TOLL_FREE: "toll_free",
        PhoneNumberType.PREMIUM_RATE: "premium_rate",
        PhoneNumberType.SHARED_COST: "shared_cost",
        PhoneNumberType.VOIP: "voip",
        PhoneNumberType.PERSONAL_NUMBER: "personal_number",
        PhoneNumberType.PAGER: "pager",
        PhoneNumberType.UAN: "uan",
        PhoneNumberType.VOICEMAIL: "voicemail",
        PhoneNumberType.UNKNOWN: "unknown",
    }

    def __init__(self, default_region: str = "RU") -> None:
        region = default_region.strip().upper()
        if region not in phonenumbers.SUPPORTED_REGIONS:
            raise ValueError(f"Неподдерживаемый код региона: {default_region}")
        self.default_region = region

    def _parse(self, phone_number: str, region: Optional[str] = None):
        if not isinstance(phone_number, str) or not phone_number.strip():
            raise ValueError("Телефонный номер должен быть непустой строкой.")

        parse_region = (region or self.default_region).strip().upper()
        if parse_region not in phonenumbers.SUPPORTED_REGIONS:
            raise ValueError(f"Неподдерживаемый код региона: {parse_region}")

        matches = list(phonenumbers.PhoneNumberMatcher(phone_number, parse_region))
        if len(matches) == 1:
            return matches[0].number
        if len(matches) > 1:
            raise ValueError("Найдено несколько телефонных номеров; укажите один.")

        try:
            return phonenumbers.parse(phone_number.strip(), parse_region)
        except NumberParseException as exc:
            raise ValueError(f"Не удалось распознать телефонный номер: {exc}") from exc

    def validate(self, phone_number: str, region: Optional[str] = None) -> bool:
        """Возвращает True, если номер существует в плане нумерации региона."""
        try:
            parsed_number = self._parse(phone_number, region)
        except ValueError:
            return False
        return phonenumbers.is_valid_number(parsed_number)

    def normalize(
        self,
        phone_number: str,
        region: Optional[str] = None,
        output_format: str = "E164",
    ) -> str:
        """Проверяет номер и возвращает его в выбранном стандартном формате."""
        format_name = output_format.strip().upper()
        if format_name not in self._FORMATS:
            supported = ", ".join(self._FORMATS)
            raise ValueError(
                f"Неподдерживаемый формат: {output_format}. Доступны: {supported}."
            )

        parsed_number = self._parse(phone_number, region)
        if not phonenumbers.is_valid_number(parsed_number):
            raise ValueError(f"Телефонный номер '{phone_number}' недействителен.")

        return phonenumbers.format_number(parsed_number, self._FORMATS[format_name])

    def analyze(
        self, phone_number: str, region: Optional[str] = None
    ) -> Dict[str, Union[str, bool]]:
        """Возвращает результат проверки и основные представления номера."""
        parsed_number = self._parse(phone_number, region)
        is_valid = phonenumbers.is_valid_number(parsed_number)
        is_possible = phonenumbers.is_possible_number(parsed_number)

        result: Dict[str, Union[str, bool]] = {
            "input": phone_number,
            "is_possible": is_possible,
            "is_valid": is_valid,
        }

        if is_valid:
            result.update(
                {
                    "e164": phonenumbers.format_number(
                        parsed_number, PhoneNumberFormat.E164
                    ),
                    "international": phonenumbers.format_number(
                        parsed_number, PhoneNumberFormat.INTERNATIONAL
                    ),
                    "national": phonenumbers.format_number(
                        parsed_number, PhoneNumberFormat.NATIONAL
                    ),
                    "region": phonenumbers.region_code_for_number(parsed_number)
                    or "unknown",
                    "number_type": self._NUMBER_TYPES.get(
                        phonenumbers.number_type(parsed_number), "unknown"
                    ),
                }
            )

        return result

    def use(self, phone_number: str) -> str:
        """Интерфейс инструмента для LLM-агента."""
        try:
            result = self.analyze(phone_number)
        except ValueError as exc:
            return f"Ошибка: {exc}"

        if not result["is_valid"]:
            possibility = "возможен" if result["is_possible"] else "невозможен"
            return (
                f"Телефонный номер '{phone_number}' недействителен "
                f"(по длине и структуре {possibility})."
            )

        return (
            f"Телефонный номер действителен. E.164: {result['e164']}; "
            f"международный формат: {result['international']}; "
            f"регион: {result['region']}; тип: {result['number_type']}."
        )
