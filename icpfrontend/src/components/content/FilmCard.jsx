import {
    Card,
    CardHeader,
    CardBody,
    CardFooter,
    Typography,
    Button,
  } from "@material-tailwind/react";
  import PropTypes from "prop-types";
   
  export function FilmCard(props) {
    const { film } = props;
    const moviePoster = `https://image.tmdb.org/t/p/w500/${film.poster_path}`;

    return (
        <Card className="">
            <CardHeader color="blue-gray" className="">
            <img
      src={moviePoster}
      alt={film.movie_title}
      className="w-full rounded-lg"
    />
  </CardHeader>
  <CardBody>
    <Typography variant="h5" color="blue-gray" className="mb-2">
      {film.movie_title}
    </Typography>
  </CardBody>
</Card>

      
    );
  }

FilmCard.propTypes = {
    film: PropTypes.object.isRequired,
};