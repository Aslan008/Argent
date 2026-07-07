"""C# structural outline so a 500-line Unity script needn't be read whole."""

from tools._helpers import _csharp_outline
from tools.file_ops import get_file_outline

CS = """using UnityEngine;

public class Player : MonoBehaviour
{
    public int health = 100;

    void Start()
    {
        Debug.Log("hi");
    }

    private void Update()
    {
        Move();
    }

    public int Damage(int amount)
    {
        return amount;
    }
}
"""


def test_outline_finds_type_and_methods():
    joined = "\n".join(_csharp_outline(CS))
    assert "class Player" in joined
    assert "Start" in joined and "Update" in joined and "Damage" in joined


def test_outline_ignores_calls_and_fields():
    joined = "\n".join(_csharp_outline(CS))
    assert "Move" not in joined          # a call, not a declaration
    assert "health" not in joined        # a field, not a method


def test_get_file_outline_dispatches_cs(tmp_path):
    f = tmp_path / "Player.cs"
    f.write_text(CS, encoding="utf-8")
    out = get_file_outline(str(f))
    assert "Outline of" in out and "class Player" in out and "Start" in out


def test_outline_rejects_unknown_extension(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello world", encoding="utf-8")
    out = get_file_outline(str(f))
    assert "only available for Python" in out
