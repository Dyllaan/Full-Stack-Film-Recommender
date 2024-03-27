import Drawer from '@mui/material/Drawer';
import PropTypes from 'prop-types';

const MyDrawer = ({ isOpen, setIsOpen, selectedSlide, posters }) => {
  // Function to close the drawer
  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setIsOpen(open);
  };

  // Access the selected poster data using selectedSlide
  const selectedPoster = posters[selectedSlide];

  return (
    <Drawer
      anchor='bottom'
      open={isOpen}
      onClose={toggleDrawer(false)}
    >
      {/* Display specific content based on the selected slide */}
      {selectedPoster ? (
        <div>
          {/* Show specific content for the selected slide here, e.g., an image or details */}
          <img src={`https://image.tmdb.org/t/p/w500${selectedPoster.poster_path}`} alt={`Selected Movie Poster`} />
          {/* You can add more details based on the selected poster data */}
        </div>
      ) : (
        <div>Select a movie to see details here.</div>
      )}
    </Drawer>
  );
};

MyDrawer.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  setIsOpen: PropTypes.func.isRequired,
  selectedSlide: PropTypes.number,
  posters: PropTypes.array.isRequired,
};

export default MyDrawer;