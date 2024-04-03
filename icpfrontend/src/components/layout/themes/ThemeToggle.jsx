import { useState } from 'react';
import PropTypes from 'prop-types';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import Button from '@mui/material/Button';
import './toggle-spin.css'

export default function ThemeToggle({ toggleTheme, theme }) {
    const [isSpinning, setIsSpinning] = useState(false);

    const handleClick = () => {
        setIsSpinning(true);
        toggleTheme();

        setTimeout(() => setIsSpinning(false), 500);
    };

    return (
        <Button onClick={handleClick} className="flex items-center gap-1" color="secondary">
            {theme === 'light' ? (
                <DarkModeIcon className={isSpinning ? 'spin-animation' : ''} />
            ) : (
                <LightModeIcon className={isSpinning ? 'spin-animation' : ''} />
            )}
        </Button>
    );
}

ThemeToggle.propTypes = {
    toggleTheme: PropTypes.func.isRequired,
    theme: PropTypes.string.isRequired,
};
