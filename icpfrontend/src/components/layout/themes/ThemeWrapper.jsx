import { ThemeProvider, createTheme } from '@mui/material/styles';
import PropTypes from 'prop-types';

// custom purple

const lightModeTheme = createTheme({
  palette: {
    primary: {
      light: '#757ce8',
      main: '#3f50b5',
      dark: '#002884',
      contrastText: '#fff',
    },
    secondary: {
      light: '#ff7961',
      main: '#121212',
      dark: '#ba000d',
      contrastText: '#000',
    },
  },
})

const darkModeTheme = createTheme({
  palette: {
    primary: {
      light: '#757ce8',
      main: '#3f50b5',
      dark: '#002884',
      contrastText: '#fff',
    },
    secondary: {
      light: '#ff7961',
      main: '#3f50b5',
      dark: '#ba000d',
      contrastText: '#000',
    },
    mode: 'dark',
  },
})


export default function ThemeWrapper({children, theme}) {
  return (
    <ThemeProvider theme={theme === 'dark' ? darkModeTheme : lightModeTheme}>
        {children}
    </ThemeProvider>
  );
}

ThemeWrapper.propTypes = {
    children: PropTypes.node,
    theme: PropTypes.string
}
