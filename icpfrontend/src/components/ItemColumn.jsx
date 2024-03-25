import PropTypes from 'prop-types';
import useFetchData from '../hooks/useFetchData';

function ItemColumn(props) {
    const {endpoint, renderItem, slice} = props;
    const { data : items } = useFetchData(endpoint);

    const getItems = () => {
        if (slice) {
            return items.slice(0, slice);
        }
        return items;
    }

    return (
        <div className="flex flex-col gap-1">
            {getItems().length > 0 && getItems().sort((a, b) => a.created - b.created).map((item, index) => (
                renderItem(item, index)
            ))}
        </div>
    );
}

ItemColumn.propTypes = {
    endpoint: PropTypes.string.isRequired,
    renderItem: PropTypes.func.isRequired,
    slice: PropTypes.number
};

export default ItemColumn;