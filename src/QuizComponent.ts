// src/QuizComponent.ts

/**
 * Component responsible for rendering the Beginner C# Multiple Choice Quiz.
 * This component assumes a framework context (e.g., Vue, React setup) 
 * where this file would be imported and utilized.
 */

interface QuizQuestion {
    id: number;
    question: string;
    options: string[];
    correctAnswerIndex: number;
}

const quizData: QuizQuestion[] = [
    {
        id: 1,
        question: "Какой ключевое слово используется для объявления целочисленной переменной в C#?",
        options: ["string count = 10;", "int count: 10;", "int count = 10;", "var int count = 10;"],
        correctAnswerIndex: 2 
    },
    {
        id: 2,
        question: "Какое ключевое слово используется для наследования классов в C#?",
        options: ["inherits", ":", "extends", "is"],
        correctAnswerIndex: 1 
    },
    {
        id: 3,
        question: "Какой модификатор используется для объявления константы, значение которой не может быть изменено?",
        options: ["static", "readonly", "const", "final"],
        correctAnswerIndex: 2 
    },
    {
        id: 4,
        question: "В контексте сравнения объектов (например, классов) в C#, какой оператор определяет равенство по значению?",
        options: ["==", "Equals()", "===", "!=="],
        correctAnswerIndex: 1
    },
    {
        id: 5,
        question: "Какой принцип программирования относится к 'Ответственность одного класса должна быть только одна'? (ООП)",
        options: ["Наследование (Inheritance)", "Полиморфизм (Polymorphism)", "Инкапсуляция (Encapsulation)", "Единственная ответственность (Single Responsibility)"],
        correctAnswerIndex: 3
    }
];

/**
 * Function to process and display the quiz.
 * In a real application, this would integrate with the UI framework's lifecycle methods.
 * @param currentQuestionIndex The index of the question to display.
 * @returns The data for the current question.
 */
export function getQuizQuestion(currentQuestionIndex: number): QuizQuestion | null {
    if (currentQuestionIndex >= 0 && currentQuestionIndex < quizData.length) {
        return quizData[currentQuestionIndex];
    }
    return null;
}

/**
 * Checks if the user's selected answer is correct.
 * @param questionId The ID of the question being answered.
 * @param selectedOptionIndex The index of the option the user clicked.
 * @returns Boolean indicating correctness.
 */
export function checkAnswer(questionId: number, selectedOptionIndex: number): boolean {
    const question = quizData.find(q => q.id === questionId);
    if (!question) return false;
    return question.correctAnswerIndex === selectedOptionIndex;
}

export { quizData };