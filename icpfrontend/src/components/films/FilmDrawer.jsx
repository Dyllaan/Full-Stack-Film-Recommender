import Drawer from '@mui/material/Drawer';
import PropTypes from 'prop-types';
import SelectedFilm from './SelectedFilm';

const FilmDrawer = ({ isOpen, setIsOpen, selectedSlide, posters }) => {
  
  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setIsOpen(open);
  };

  const selectedFilm = posters[selectedSlide];

  return (
    <Drawer
      anchor='bottom'
      open={isOpen}
      onClose={toggleDrawer(false)}
      sx={
        {
          '& .MuiDrawer-paper': {
            width: '100%',
            maxHeight: '60vh',
          },
          /**Darkens the bg */
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
        }
      }
    >
      {/* Display specific content based on the selected film */}
      {selectedFilm? (
        <SelectedFilm film={selectedFilm} />
      ) : (
        <div>Select a movie to see details here.</div>
      )}
    </Drawer>
  );
};

FilmDrawer.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  setIsOpen: PropTypes.func.isRequired,
  selectedSlide: PropTypes.number,
  posters: PropTypes.array.isRequired,
};

export default FilmDrawer;