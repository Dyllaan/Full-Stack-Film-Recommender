import {useEffect, useState} from 'react';
import Rating from '@mui/material/Rating';
import StarIcon from '@mui/icons-material/Star';
import  PropTypes from 'prop-types';

export default function RateFilm(props) {
  const { film, handleRatingSubmission } = props;
  const [value, setValue] = useState(0);

  useEffect(() => {
    if(value !== 0) {
      handleRatingSubmission(value, film);
    }
  }, [value]);

  return (
    <div>
      <Rating name="half-rating" defaultValue={2.5} precision={0.5}
        onChange={(event, newValue) => {
          setValue(newValue);
        }}
        emptyIcon={<StarIcon style={{ opacity: 0.55, color:'red'}} fontSize="inherit" />}
      />
    </div>
  );
}

RateFilm.propTypes = {
  film: PropTypes.object.isRequired,
  handleRatingSubmission: PropTypes.func.isRequired,
};