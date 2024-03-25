/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontSize: {
        xs: '0.7rem',
        sm: '0.8rem',
        base: '0.9rem',
        lg: '1.35rem',
        xl: '1.5rem',
        '2xl': '1.8rem',
        '3xl': '2rem',
        '4xl': '2.5rem',
        '5xl': '3rem',
      },
      scale: {
        '1025': '1.025',
      },
      transitionDuration: {
        '400': '400ms',
      },
      backgroundImage: {
        'education': "url('/education.jpg')",
      },
      backgroundColor: {
        'custom': '#0F141C',      
        'deep-blue': '#08052b',
        'deep-purple': '#0f0212',
      },
      borderColor: {
        'custom': '#181f66',
      },
    },
  },
  plugins: [],
}

