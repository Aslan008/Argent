"""Tests for native C/C++ and NTSTATUS crash diagnostics in command_diagnostics.py."""

import pytest
from src.agent.command_diagnostics import diagnose_command_error


class TestNativeCrashDiagnostics:
    """Verifies recognition of NTSTATUS crash codes and POSIX signals."""

    def test_status_access_violation_diagnosed(self):
        hint = diagnose_command_error("solver.exe", "", -1073741819)
        assert hint is not None
        assert "STATUS_ACCESS_VIOLATION" in hint
        assert "Segmentation Fault" in hint
        assert "AddressSanitizer" in hint

    def test_status_stack_overflow_diagnosed(self):
        hint = diagnose_command_error("./recursive", "", -1073741571)
        assert hint is not None
        assert "STATUS_STACK_OVERFLOW" in hint
        assert "recursion" in hint

    def test_posix_sigsegv_diagnosed(self):
        hint = diagnose_command_error("./bin", "", 139)
        assert hint is not None
        assert "SIGSEGV" in hint

    def test_posix_sigabrt_diagnosed(self):
        hint = diagnose_command_error("./bin", "", 134)
        assert hint is not None
        assert "SIGABRT" in hint


class TestCppCompilerDiagnostics:
    """Verifies recognition of C/C++ linker and compiler errors."""

    def test_msvc_unresolved_external_lnk2019(self):
        output = "Physics.obj : error LNK2019: unresolved external symbol solve_physics referenced in function main"
        hint = diagnose_command_error("cmake --build .", output, 1)
        assert hint is not None
        assert "Unresolved External Symbol" in hint
        assert "CMakeLists.txt" in hint

    def test_gcc_clang_undefined_reference(self):
        output = "/usr/bin/ld: main.cpp:(.text+0x15): undefined reference to `calculate_trajectory()'"
        hint = diagnose_command_error("g++ main.cpp", output, 1)
        assert hint is not None
        assert "Unresolved External Symbol" in hint

    def test_include_not_found(self):
        output = "Physics.cpp:1:10: fatal error: glm/glm.hpp: No such file or directory"
        hint = diagnose_command_error("clang++ Physics.cpp", output, 1)
        assert hint is not None
        assert "Include Not Found" in hint
        assert "target_include_directories" in hint

    def test_cmake_error_diagnosed(self):
        output = "CMake Error at CMakeLists.txt:10 (find_package): Could not find package OpenSSL"
        hint = diagnose_command_error("cmake -B build", output, 1)
        assert hint is not None
        assert "CMake Configuration Error" in hint
