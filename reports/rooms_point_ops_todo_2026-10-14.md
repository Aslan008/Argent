# Точечные операции над комнатой для основной LLM (2026-10-14)

Цель: модель может создать комнату, добавить/удалить/переименовать блок, соединить/разъединить,
менять любой параметр блока, стрелки и комнаты — без переписывания комнаты целиком.
Сохранение — только после подтверждения человека, как у build/edit.

- [x] P1. src/rooms/patch.py: apply_ops(room_dict, ops) — add_node, insert_node, remove_node, rename_node, set_param, connect, disconnect, set_edge, set_room
- [x] P2. tool.py: действия `patch` и `create` (атомарно, validate_room + harden + diff + показ схемы + подтверждение + save_as)
- [x] P3. tools/schemas.py + tools/rooms_tools.py: схема (ops, save_as), описание для модели
- [x] P4. plan_mode: patch/create блокируются в режиме плана как build/edit
- [x] P5. Тесты tests/test_rooms_patch.py (операции, атомарность, harden, starter → _custom, отказ человека, лимиты)
- [x] P6. Живая проверка: реальный вызов инструмента rooms(patch) на копии комнаты
- [x] P7. Прогон rooms+automation+phantom_tools, отчёт (раздел 9) и память
