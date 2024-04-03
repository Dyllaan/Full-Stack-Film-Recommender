import useTMDb from "../../hooks/useTMDb";
import PropTypes from 'prop-types';
import { useEffect } from "react";
import { Link } from "react-router-dom";
import { Button, Paper, Box } from "@mui/material";

/**
 * As MovieLens misses a lot of data, we can use the TMDb API to get more information about the selected film.
 */

export default function SelectedFilm({ film }) {

    const { data, loading } = useTMDb(film.tmdb_id);

    useEffect(() => {
        console.log(data);
    }, [data]);

    if(loading) {
        return;
    }

    return (
        <Box className="flex flex-row p-8 gap-16">
            <Paper className="mx-auto w-[50vw] flex flex-col rounded-lg p-2 items-center">
                <div className="flex flex-col">
                    <div className="bg-indigo-950 p-4 rounded-full">
                        <h2 className="text-indigo-500">{data.title}</h2>
                    </div>
                </div>
                <div className="p-4">
                    <p className="text-lg">{data.overview}</p>
                </div>
                <div>
                    <Button variant="outlined"><Link to={`/film/${film.movie_slug}`}>See more</Link></Button>
                </div>
          </Paper>
        </Box>
    );
}

SelectedFilm.propTypes = {
    film: PropTypes.object.isRequired,
};