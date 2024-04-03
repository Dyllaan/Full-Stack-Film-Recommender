/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      scale: {
        '1025': '1.025',
      },
      transitionDuration: {
        '400': '400ms',
      },
      backgroundImage: {
        'placeholder': "url('/placeholder.png')",
      },
      backgroundColor: {
        'custom': '#0F141C',      
      },
      textColor: {
        'off-white': '#FAF9F6',
      },
      borderColor: {
        'custom': '#181f66',
      },
    },
  },
  plugins: [],
}